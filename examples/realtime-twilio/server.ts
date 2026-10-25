import Fastify from 'fastify';
import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import twilio from 'twilio';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import { RealtimeAgent, RealtimeSession, tool } from '@openai/agents/realtime';
import { TwilioRealtimeTransportLayer } from '@openai/agents-extensions';
import { hostedMcpTool } from '@openai/agents';
import { z } from 'zod';
import process from 'node:process';

export type TwilioServerConfig = {
  openAiApiKey: string;
  twilioAuthToken: string;
  publicBaseUrl: string;
};

export function buildServer(config: TwilioServerConfig): FastifyInstance {
  if (!config.openAiApiKey || !config.twilioAuthToken) {
    throw new Error('OPENAI_API_KEY and TWILIO_AUTH_TOKEN are required.');
  }
  let publicUrl: URL;
  try {
    publicUrl = new URL(config.publicBaseUrl);
  } catch {
    throw new Error('TWILIO_PUBLIC_BASE_URL must be a valid HTTPS origin.');
  }
  if (
    publicUrl.protocol !== 'https:' ||
    publicUrl.username ||
    publicUrl.password ||
    publicUrl.pathname !== '/' ||
    publicUrl.search ||
    publicUrl.hash
  ) {
    throw new Error(
      'TWILIO_PUBLIC_BASE_URL must be an HTTPS origin without credentials, a path, query, or fragment.',
    );
  }
  const publicOrigin = publicUrl.origin;
  const streamUrl =
    publicOrigin.replace('https://', 'wss://') + '/media-stream';
  const signature = (request: FastifyRequest) => {
    const value = request.headers['x-twilio-signature'];
    return typeof value === 'string' ? value : '';
  };

  const debugRealtimeEventTypes = new Set([
    'session.created',
    'session.updated',
    'input_audio_buffer.speech_started',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.committed',
    'conversation.item.input_audio_transcription.completed',
    'response.created',
    'response.output_audio.done',
    'response.done',
    'error',
  ]);

  function decodeMuLawSample(value: number): number {
    const sample = ~value & 0xff;
    const sign = sample & 0x80;
    const exponent = (sample >> 4) & 0x07;
    const mantissa = sample & 0x0f;
    const magnitude = ((mantissa << 3) + 0x84) * 2 ** exponent - 0x84;
    return sign === 0 ? magnitude : -magnitude;
  }

  // State for the call currently being handled. Ed Vertex receives one
  // call at a time on this line, so a single holder is enough: the
  // /incoming-call webhook fills it in when a call starts and the
  // save_service_request tool reads it when the agent saves the request.
  const pendingCall = {
    callerPhone: '',
    startedAt: '',
    summarySent: false,
  };

  // Sends the end-of-call service summary to Eduardo by email (Resend).
  // Recipient defaults to the Ed Vertex Gmail; override with
  // SUMMARY_EMAIL_TO. Sender uses the Resend onboarding address until a
  // domain is verified. If RESEND_API_KEY is missing, it just logs.
  async function sendSummaryEmail(lines: string[]): Promise<boolean> {
    const apiKey = process.env.RESEND_API_KEY;
    const to = process.env.SUMMARY_EMAIL_TO || 'edvertex.morales@gmail.com';
    if (!apiKey) {
      console.log('RESEND_API_KEY not set; summary email skipped.');
      return false;
    }
    try {
      const response = await fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          from: 'Agente Ed Vertex <onboarding@resend.dev>',
          to: [to],
          subject: lines[0],
          text: lines.slice(1).join('\n'),
        }),
      });
      if (!response.ok) {
        console.error(
          'Summary email failed:',
          response.status,
          await response.text(),
        );
        return false;
      }
      console.log('Summary email sent to', to);
      return true;
    } catch (error) {
      console.error('Summary email error.', error);
      return false;
    }
  }

  // Initialize Fastify.
  const fastify = Fastify();
  fastify.register(fastifyFormBody);
  fastify.register(fastifyWs);

  const weatherTool = tool({
    name: 'weather',
    description: 'Get the weather in a given location.',
    parameters: z.object({
      location: z.string(),
    }),
    execute: async ({ location }: { location: string }) => {
      return `The weather in ${location} is sunny.`;
    },
  });

  const secretTool = tool({
    name: 'secret',
    description: 'A secret tool to tell the special number.',
    parameters: z.object({
      question: z
        .string()
        .describe(
          'The question to ask the secret tool; mainly about the special number.',
        ),
    }),
    execute: async ({ question }: { question: string }) => {
      return `The answer to ${question} is 42.`;
    },
    needsApproval: true,
  });

  // Saves the service request collected during the call and emails the
  // summary to Eduardo. The agent must call this once, right before
  // giving the caller the final verbal summary.
  const saveServiceRequestTool = tool({
    name: 'save_service_request',
    description:
      'Save the service request collected during this call and email the summary to the Ed Vertex office. Call this exactly once, after collecting the customer information and before giving the final summary. For anything the caller did not provide, pass "No proporcionado".',
    parameters: z.object({
      fullName: z.string().describe('Full customer name.'),
      callbackPhone: z
        .string()
        .describe('Callback phone number confirmed with the caller.'),
      serviceAddress: z
        .string()
        .describe('Complete service address exactly as confirmed.'),
      zipCode: z.string().describe('ZIP code of the service address.'),
      propertyType: z
        .string()
        .describe('Residential or commercial property.'),
      condoOrHoa: z
        .string()
        .describe('Whether it is a condominium or HOA property, if known.'),
      serviceNeeded: z
        .string()
        .describe('Brief description of the HVAC service needed.'),
      equipmentLocation: z
        .string()
        .describe('Where the HVAC equipment is located.'),
      accessInstructions: z
        .string()
        .describe('Access instructions: gate, guard, code, parking.'),
      email: z.string().describe('Customer email address.'),
      preferredDayTime: z
        .string()
        .describe(
          'Preferred day and time for service (a preference, not a confirmed appointment).',
        ),
      callLanguage: z
        .string()
        .describe('Language the call was held in: Spanish or English.'),
    }),
    execute: async (input: {
      fullName: string;
      callbackPhone: string;
      serviceAddress: string;
      zipCode: string;
      propertyType: string;
      condoOrHoa: string;
      serviceNeeded: string;
      equipmentLocation: string;
      accessInstructions: string;
      email: string;
      preferredDayTime: string;
      callLanguage: string;
    }) => {
      if (pendingCall.summarySent) {
        return 'The service request was already saved for this call. Give the final summary and close the call.';
      }
      pendingCall.summarySent = true;
      const when = new Intl.DateTimeFormat('es-US', {
        timeZone: 'America/New_York',
        dateStyle: 'full',
        timeStyle: 'short',
      }).format(new Date());
      await sendSummaryEmail([
        `Nueva solicitud de servicio — ${input.fullName} — Agente de voz Ed Vertex`,
        `Fecha y hora de la llamada: ${when} (hora del Este)`,
        `Teléfono de la llamada (Twilio): ${pendingCall.callerPhone || 'No disponible'}`,
        '',
        `Nombre: ${input.fullName}`,
        `Teléfono de contacto: ${input.callbackPhone}`,
        `Dirección de servicio: ${input.serviceAddress}`,
        `ZIP: ${input.zipCode}`,
        `Tipo de propiedad: ${input.propertyType}`,
        `Condominio / HOA: ${input.condoOrHoa}`,
        `Servicio solicitado: ${input.serviceNeeded}`,
        `Ubicación del equipo: ${input.equipmentLocation}`,
        `Acceso: ${input.accessInstructions}`,
        `Correo del cliente: ${input.email}`,
        `Horario preferido: ${input.preferredDayTime}`,
        `Idioma de la llamada: ${input.callLanguage}`,
        '',
        'Resumen generado por tu agente de voz de Ed Vertex. El horario es una preferencia del cliente: confirma disponibilidad antes de comprometer la visita.',
      ]);
      return 'Service request saved. Now give the caller the final summary and close the call politely.';
    },
  });

  const agent = new RealtimeAgent({
    name: 'Ed Vertex Voice Agent',
    instructions: `
You are the bilingual virtual receptionist for Ed Vertex LLC, a professional HVAC company serving South Florida.

LANGUAGE
Speak naturally in the language used by the caller.
If the caller speaks Spanish, remain completely in Spanish.
If the caller speaks English, remain completely in English.

OPENING
As soon as the call is connected, greet the caller immediately.
You speak FIRST. Do not wait for the caller to say hello first.
Greet with this exact bilingual greeting:
"Thank you for calling Ed Vertex. How can we help you with your air conditioning system? Si prefiere español, con gusto le atendemos en español."
After the greeting, continue the entire conversation in the language the caller answers in.

CONVERSATION STYLE
Be warm, professional, concise, calm, and conversational.
Ask only ONE question at a time.
Never explain your internal process.
Do not repeatedly thank the caller after every answer.
Do not repeat information that has already been clearly provided and confirmed.

CUSTOMER INFORMATION
Collect naturally, when relevant:
- Full customer name
- Callback phone number
- Complete service address
- ZIP code
- Residential or commercial property
- Brief description of the HVAC service needed
- Whether the property is a condominium or HOA
- Whether contractor documents or a COI may be required
- Location of the HVAC equipment, such as closet, attic, garage, roof, mechanical room, or exterior
- Access instructions, such as gate, guard, code, parking, or equipment access
- Email address
- Preferred day and time for service

NAME
When the caller provides their name, confirm it once.
After the name has been confirmed, do not ask for it again.

PHONE NUMBER
When confirming a U.S. phone number, group it naturally as:
XXX-XXX-XXXX.
Confirm it once and then continue.

SERVICE ADDRESS
Treat the address as critical information.
Do not guess, reconstruct, substitute, or invent any portion of an address.
If one portion is unclear, ask the caller to repeat ONLY that portion.
Once all parts are understood, repeat the COMPLETE address back once and ask:
"¿Es correcto?"
or in English:
"Is that correct?"
Do not continue until the caller confirms or corrects the address.

SERVICE TIME
The caller may provide a preferred day and time, but you do NOT have access to the live Ed Vertex calendar yet.
Never tell the caller that an appointment has been booked, scheduled, confirmed, or reserved.
Never say that a technician will arrive at that time.
Instead say:
"Anoto [day/time] como su horario preferido. Ed Vertex confirmará la disponibilidad."
In English:
"I've noted [day/time] as your preferred service time. Ed Vertex will confirm availability."

HVAC
Do not quote repair or replacement prices.
Do not make a definitive HVAC diagnosis by phone.
Do not pressure callers to purchase equipment.
Do not invent technical information.
Ed Vertex performs a professional evaluation before recommending repair or replacement.

EQUIPMENT LOCATION
Ask naturally where the HVAC equipment is located.
Do not describe a closet or attic as being outdoors.
Examples of locations include closet, attic, garage, roof, mechanical room, or exterior.

...
EQUIPMENT LOCATION
Ask naturally where the HVAC equipment is located.
Do not describe a closet or attic as being outdoors.
Examples of locations include closet, attic, garage, roof, mechanical room, or exterior.

BRAND NAME — CRITICAL

The company name is "Ed Vertex".

When speaking the company name, pronounce it clearly as:
"Ed VER-teks"

Never say:
- Ed Burkett
- Ed Vortex
- Edvertex
- EverTech
- Edwardtex

LANGUAGE LOCK — CRITICAL

Once the caller chooses Spanish, speak ONLY Spanish for the rest of the call unless the caller explicitly asks to switch languages.

Do not use English filler phrases in a Spanish conversation, including:
- Okay
- Thanks
- Got it
- Let me
- Perfect

Once the caller chooses English, remain completely in English.

EMAIL — REQUIRED ATTEMPT

Before asking for preferred service day and time, always ask for the customer's email address.

Say in Spanish:
"¿Cuál es el mejor correo electrónico para mantener su información de servicio actualizada?"

Say in English:
"What is the best email address to keep your service information updated?"

Repeat the email once for confirmation.
Never guess an email address.
If the customer declines to provide one, continue normally.

APPOINTMENT WORDING

Never say:
"Ed Vertex confirms availability."

Say:
"Ed Vertex will confirm availability."

Spanish:
"Ed Vertex confirmará la disponibilidad."

ADDRESS NUMBERS — CRITICAL

Street numbers and street names must be preserved EXACTLY as spoken by the caller.

Never convert, shorten, reinterpret, or guess an address number or ordinal.

Examples:
105th must NEVER become 10th.
105th must NEVER become 105.
61st must NEVER become 6th.

If any street number, ordinal, direction, suffix, unit, or ZIP code is unclear, ask the caller to repeat ONLY that specific part.

When confirming the address, repeat exactly what the caller provided.

Do not normalize an address unless the caller explicitly confirms the normalized version.

NO DUPLICATE QUESTIONS

Before asking a question, check whether the caller already provided that information.
If it has already been clearly provided, do not ask for it again.

CLOSING

After giving the final summary, do not ask additional optional questions.
End the call naturally unless the caller continues the conversation.

SAVING THE REQUEST — REQUIRED

When you have collected the customer's information, and BEFORE you give the final summary, call the save_service_request tool exactly once with everything the caller provided: name, callback phone, complete service address, ZIP code, property type, condominium or HOA if known, service needed, equipment location, access instructions, email, preferred day and time, and the language of the call.
For anything the caller did not provide, pass "No proporcionado".
Never mention the tool, the save, or any email to the caller.
After the tool returns, give the final summary and close the call as usual.

ENDING
Before ending, give a short summary of the service request using only information the caller actually provided.
Clearly state that the requested day and time are preferences pending confirmation.
Do not claim that an email, text message, confirmation, or appointment has been sent or created unless a tool actually performed that action.

`,
  tools: [saveServiceRequestTool],
  });

  // Root route.
  fastify.get('/', async (_request: FastifyRequest, reply: FastifyReply) => {
    reply.send({ message: 'Twilio Media Stream Server is running!' });
  });

  // Authenticate call webhooks before returning call instructions.
  fastify.route({
    method: ['GET', 'POST'],
    url: '/incoming-call',
    exposeHeadRoute: false,
    onRequest: async (request, reply) => {
      if (!signature(request)) {
        return reply.code(403).send('Invalid Twilio signature.');
      }
      if (
        request.method === 'POST' &&
        request.headers['content-type']?.split(';')[0].trim().toLowerCase() !==
          'application/x-www-form-urlencoded'
      ) {
        return reply.code(415).send('Expected a form-encoded Twilio webhook.');
      }
    },
    preValidation: async (request, reply) => {
      const rawUrl = request.raw.url ?? '';
      const queryStart = rawUrl.indexOf('?');
      const query = queryStart < 0 ? '' : rawUrl.slice(queryStart);
      const url = publicOrigin + '/incoming-call' + query;
      const params =
        request.method === 'POST'
          ? (request.body as Record<string, string | string[]>)
          : {};
      if (
        !twilio.validateRequest(
          config.twilioAuthToken,
          signature(request),
          url,
          params,
        )
      ) {
        return reply.code(403).send('Invalid Twilio signature.');
      }
    },
    handler: async (request, reply) => {
      // Remember who is calling so the end-of-call summary email can
      // include the real caller number reported by Twilio.
      const body = (request.body ?? {}) as Record<string, unknown>;
      pendingCall.callerPhone =
        typeof body.From === 'string' ? body.From : '';
      pendingCall.startedAt = new Date().toISOString();
      pendingCall.summarySent = false;
      const response = new twilio.twiml.VoiceResponse();
      response.connect().stream({ url: streamUrl });
      return reply.type('text/xml').send(response.toString());
    },
  });

  // WebSocket route for media-stream.
  fastify.register(async (scopedFastify: FastifyInstance) => {
    scopedFastify.get(
      '/media-stream',
      {
        websocket: true,
        preValidation: async (request, reply) => {
          const signed = signature(request);
          if (
            !signed ||
            request.raw.url?.includes('?') ||
            !(
              twilio.validateRequest(
                config.twilioAuthToken,
                signed,
                streamUrl,
                {},
              ) ||
              twilio.validateRequest(
                config.twilioAuthToken,
                signed,
                streamUrl + '/',
                {},
              )
            )
          ) {
            return reply.code(403).send('Invalid Twilio signature.');
          }
        },
      },
      async (connection: any) => {
        let twilioTransportLayer: TwilioRealtimeTransportLayer | undefined;
        let session: RealtimeSession | undefined;
        try {
          twilioTransportLayer = new TwilioRealtimeTransportLayer({
            twilioWebSocket: connection,
          });

          session = new RealtimeSession(agent, {
            transport: twilioTransportLayer,
            model: 'gpt-realtime-2.1',
            config: {
              audio: {
                input: {
                  transcription: {
                    model: 'gpt-4o-mini-transcribe',
                  },
                },
                output: {
                  voice: 'verse',
                },
              },
            },
          });

          const connectedSession = session;
          session.on('error', (error: unknown) => {
            console.error('Realtime session error.', error);
          });

          // Log the full conversation transcript (caller + agent) so each
          // call can be reviewed afterwards in the Render logs.
          session.on(
            'transport_event',
            (event: { type: string; transcript?: string }) => {
              if (
                event.type ===
                  'conversation.item.input_audio_transcription.completed' &&
                typeof event.transcript === 'string' &&
                event.transcript.trim()
              ) {
                console.log(`TRANSCRIPT CLIENTE: ${event.transcript}`);
              }
              if (
                event.type === 'response.output_audio_transcript.done' &&
                typeof event.transcript === 'string' &&
                event.transcript.trim()
              ) {
                console.log(`TRANSCRIPT AGENTE: ${event.transcript}`);
              }
            },
          );

          if (process.env.DEBUG) {
            connection.addEventListener('close', (event: any) => {
              console.log(
                'Twilio WebSocket closed:',
                JSON.stringify({
                  code: event.code,
                  reason: event.reason ? String(event.reason) : undefined,
                }),
              );
            });
            let inputSampleCount = 0;
            let inputSquareSum = 0;
            let inputPeak = 0;
            let mediaFrameCount = 0;
            let mediaByteCount = 0;
            let minimumMediaBytes = Number.POSITIVE_INFINITY;
            let maximumMediaBytes = 0;
            let previousMediaTimestamp: number | undefined;
            let previousMediaArrivalTime: number | undefined;
            let previousMediaChunk: string | undefined;
            const mediaTracks = new Set<string>();

            session.on(
              'transport_event',
              (event: {
                type: string;
                message?: {
                  event?: string;
                  start?: {
                    mediaFormat?: {
                      encoding?: string;
                      sampleRate?: number;
                      channels?: number;
                    };
                  };
                  media?: {
                    payload?: string;
                    track?: string;
                    chunk?: string;
                    timestamp?: string;
                  };
                };
                session?: {
                  audio?: {
                    input?: {
                      format?: unknown;
                      noise_reduction?: unknown;
                      turn_detection?: unknown;
                    };
                  };
                };
                response?: {
                  status?: string;
                  status_details?: unknown;
                };
              }) => {
                if (debugRealtimeEventTypes.has(event.type)) {
                  console.log(`Realtime transport event: ${event.type}`);
                }
                if (
                  event.type === 'twilio_message' &&
                  event.message?.event === 'start'
                ) {
                  console.log(
                    'Twilio media format:',
                    JSON.stringify(event.message.start?.mediaFormat),
                  );
                }
                if (
                  event.type === 'twilio_message' &&
                  event.message?.event === 'stop'
                ) {
                  console.log('Twilio media stream stopped.');
                }
                if (
                  event.type === 'twilio_message' &&
                  event.message?.event === 'media' &&
                  typeof event.message.media?.payload === 'string'
                ) {
                  const payload = event.message.media.payload;
                  const media = event.message.media;
                  const audio = Buffer.from(payload, 'base64');
                  const arrivalTime = Date.now();
                  const mediaTimestamp = Number(media.timestamp);
                  const arrivalGap =
                    previousMediaArrivalTime === undefined
                      ? undefined
                      : arrivalTime - previousMediaArrivalTime;
                  const timestampGap =
                    previousMediaTimestamp === undefined ||
                    !Number.isFinite(mediaTimestamp)
                      ? undefined
                      : mediaTimestamp - previousMediaTimestamp;

                  if (typeof media.track === 'string') {
                    mediaTracks.add(media.track);
                  }
                  mediaFrameCount += 1;
                  mediaByteCount += audio.byteLength;
                  minimumMediaBytes = Math.min(
                    minimumMediaBytes,
                    audio.byteLength,
                  );
                  maximumMediaBytes = Math.max(
                    maximumMediaBytes,
                    audio.byteLength,
                  );

                  if (
                    (arrivalGap !== undefined && arrivalGap > 100) ||
                    (timestampGap !== undefined && timestampGap > 100)
                  ) {
                    console.log(
                      'Twilio media gap:',
                      JSON.stringify({
                        arrivalMs: arrivalGap,
                        timestampMs: timestampGap,
                        previousChunk: previousMediaChunk,
                        chunk: media.chunk,
                      }),
                    );
                  }

                  if (mediaFrameCount % 50 === 0) {
                    console.log(
                      'Twilio media frames:',
                      JSON.stringify({
                        frames: mediaFrameCount,
                        averageBytes: Math.round(
                          mediaByteCount / mediaFrameCount,
                        ),
                        minimumBytes: minimumMediaBytes,
                        maximumBytes: maximumMediaBytes,
                        tracks: [...mediaTracks],
                        latestTimestamp: Number.isFinite(mediaTimestamp)
                          ? mediaTimestamp
                          : media.timestamp,
                      }),
                    );
                  }

                  previousMediaArrivalTime = arrivalTime;
                  previousMediaTimestamp = Number.isFinite(mediaTimestamp)
                    ? mediaTimestamp
                    : undefined;
                  previousMediaChunk = media.chunk;

                  for (const value of audio) {
                    const sample = decodeMuLawSample(value);
                    inputSampleCount += 1;
                    inputSquareSum += sample * sample;
                    inputPeak = Math.max(inputPeak, Math.abs(sample));
                  }
                  if (inputSampleCount >= 8000) {
                    const rms = Math.sqrt(inputSquareSum / inputSampleCount);
                    const rmsDbfs =
                      rms === 0 ? -96 : 20 * Math.log10(rms / 32768);
                    const peakDbfs =
                      inputPeak === 0
                        ? -96
                        : 20 * Math.log10(inputPeak / 32768);
                    console.log(
                      `Twilio input level: rms=${rmsDbfs.toFixed(1)} dBFS peak=${peakDbfs.toFixed(1)} dBFS`,
                    );
                    inputSampleCount = 0;
                    inputSquareSum = 0;
                    inputPeak = 0;
                  }
                }
                if (event.type === 'session.updated') {
                  const inputAudio = event.session?.audio?.input;
                  console.log(
                    'Realtime input configuration:',
                    JSON.stringify({
                      format: inputAudio?.format,
                      noiseReduction: inputAudio?.noise_reduction,
                      turnDetection: inputAudio?.turn_detection,
                    }),
                  );
                }
                if (event.type === 'response.done') {
                  console.log(
                    'Realtime response completed:',
                    JSON.stringify({
                      status: event.response?.status,
                      statusDetails: event.response?.status_details,
                    }),
                  );
                }
              },
            );
            session.on('audio_start', () => {
              console.log('Realtime audio started.');
            });
            session.on('audio_stopped', () => {
              console.log('Realtime audio stopped.');
            });
            session.on('audio_interrupted', () => {
              console.log('Realtime audio interrupted.');
            });
          }

          session.on('mcp_tools_changed', (tools: { name: string }[]) => {
            const toolNames = tools.map((tool) => tool.name).join(', ');
            console.log(`Available MCP tools: ${toolNames || 'None'}`);
          });

          session.on(
            'tool_approval_requested',
            (_context: unknown, _agent: unknown, approvalRequest: any) => {
              console.log(
                `Approving tool call for ${approvalRequest.approvalItem.rawItem.name}.`,
              );
              connectedSession
                .approve(approvalRequest.approvalItem)
                .catch((error: unknown) =>
                  console.error('Failed to approve tool call.', error),
                );
            },
          );

          session.on(
            'mcp_tool_call_completed',
            (_context: unknown, _agent: unknown, toolCall: unknown) => {
              console.log('MCP tool call completed.', toolCall);
            },
          );

          await session.connect({
            apiKey: config.openAiApiKey,
          });
          console.log('Connected to the OpenAI Realtime API');
          // The agent speaks first: trigger the opening greeting as soon as
          // the session is connected so the caller hears the agent greet
          // them instead of silence.
          session.sendMessage(
            'The call has just connected. Greet the caller now with your bilingual opening greeting.',
          );
        } catch (error) {
  console.error('Twilio session startup failed.', error);
          try {
            if (session) session.close();
            else twilioTransportLayer?.close();
          } catch {
            console.error('Twilio session cleanup failed.');
          } finally {
            connection.close(1011, 'Session startup failed.');
          }
        }
      },
    );
  });

  return fastify;
}
