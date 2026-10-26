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
import { readFileSync } from 'node:fs';

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

  // Registered-customer directory for caller recognition (Level 2).
  // Loaded once at startup from a Render Secret File (never from the
  // code repository): /etc/secrets/clientes.json maps a caller's phone
  // in E.164 format (+1XXXXXXXXXX) to their record exported from
  // Housecall Pro. If the file is missing or invalid, recognition is
  // simply off and every call runs the normal intake.
  type KnownClient = {
    name: string;
    email: string;
    address: string;
    extraAddresses: number;
    company: string;
    doNotService: boolean;
  };
  let clientDirectory: Record<string, KnownClient> = {};
  try {
    const rawDirectory = readFileSync(
      process.env.CLIENTS_FILE_PATH || '/etc/secrets/clientes.json',
      'utf8',
    );
    const parsedDirectory: unknown = JSON.parse(rawDirectory);
    if (
      parsedDirectory &&
      typeof parsedDirectory === 'object' &&
      !Array.isArray(parsedDirectory)
    ) {
      clientDirectory = parsedDirectory as Record<string, KnownClient>;
    }
    console.log(
      `Client directory loaded: ${Object.keys(clientDirectory).length} numbers.`,
    );
  } catch {
    console.log(
      'Client directory file not found; caller recognition is off.',
    );
  }

  // Extra instructions appended to the agent for a recognized caller.
  function knownCallerBlock(client: KnownClient): string {
    const firstName = client.name.split(' ')[0] || client.name;
    let block = `

REGISTERED CUSTOMER RECOGNIZED BY PHONE NUMBER
The number this caller is calling from matches a registered Ed Vertex customer. This IS a returning customer.
Name on file: ${client.name}
Email on file: ${client.email || '(none on file)'}
Registered service address: ${client.address || '(none on file)'}`;
    if (client.extraAddresses > 0) {
      block += `
This customer has ${client.extraAddresses} additional address(es) on file. If the service is for a different property than the registered address above, ask for that address with the normal procedure.`;
    }
    if (client.doNotService) {
      block += `
OFFICE NOTE — never mention this to the caller: this record is flagged "do not service" in the office system. Take the request completely normally; the office reviews every request before scheduling.`;
    }
    block += `

How to run this call:
- Complete the standard two-sentence greeting first. Then, as soon as the caller answers and the language of the call is set, greet them by name. Spanish: "Hola, ${firstName}. Qué bueno tenerlo de vuelta." English: "Hello, ${firstName}, welcome back."
- Do NOT ask for their name, and do NOT ask whether they are new or returning — you already know. When saving the request, customerType is "Ya es cliente".
- Callback phone: tell them you can see the number they are calling from, read it back grouped XXX-XXX-XXXX per the PHONE rules, and confirm it is still the best callback number. If they give a different callback number, run the normal phone procedure for the new one.
- Service address: confirm the registered address instead of asking for one from scratch. Spanish: "¿El servicio es para su dirección registrada en ${client.address}?" English: "Is the service for your registered address at ${client.address}?" If they confirm, that address is the service address — read it back once exactly as the SERVICE ADDRESS rules require (in Spanish calls, the house number and the street number digit by digit in Spanish). If the service is for a different address, switch to the normal address procedure.
- Email: if an email is on file, confirm it instead of asking them to dictate one: read it back spelled out exactly per the EMAIL rules and ask whether it is still correct. If they give a new email, run the normal email procedure for the new one. If no email is on file, ask for it normally.
- Everything else follows the normal intake order: service needed, residential or commercial, condominium or HOA, equipment location, access instructions, preferred day and time.
- If the caller says they are NOT the customer on file (for example, they are calling from someone else's phone), apologize briefly, ignore everything in this section, and run the completely normal intake as a new caller.`;
    return block;
  }

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
      'Save the service request collected during this call and email the summary to the Ed Vertex office. Call this exactly once, after collecting the customer information and before giving the final summary. For anything the caller did not provide, pass "No proporcionado". When saving, write the service address in standard written form — digits for numbers and complete street names, for example "6155 Northwest 105th Court, Apartment 3110, Doral, FL 33178" — never in spoken number words.',
    parameters: z.object({
      fullName: z.string().describe('Full customer name.'),
      customerType: z
        .string()
        .describe(
          'Whether the caller said they are already an Ed Vertex customer ("Ya es cliente") or a first-time caller ("Cliente nuevo").',
        ),
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
      askedForEduardo: z
        .string()
        .describe(
          'If the caller asked to speak with Eduardo (the owner), write "Sí" followed by the reason, for example "Sí — quiere confirmar el presupuesto de su instalación". If they did not ask, write "No".',
        ),
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
      askedForEduardo: string;
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
        `Tipo de cliente: ${input.customerType}`,
        ...(input.askedForEduardo && input.askedForEduardo !== 'No'
          ? [`⚠️ PIDIÓ HABLAR CON EDUARDO: ${input.askedForEduardo}`]
          : []),
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
The greeting has TWO mandatory sentences. Say BOTH, in this order, word for word:
Sentence 1: "Thank you for calling Ed Vertex."
Sentence 2: "Si prefiere español, con gusto le atendemos en español."
The Spanish sentence is NEVER optional: a caller who only speaks Spanish must hear it, or they will hang up. Never skip it, never replace it with another English sentence, and never ask a question before completing both sentences.
After BOTH sentences, ask: "How can we help you with your air conditioning system?"
Then STOP and wait. Say nothing else until the caller speaks — never add a second greeting or an extra question in either language.
After the greeting, continue the entire conversation in the language the caller answers in.
Greet ONLY ONCE per call. If your greeting is interrupted, NEVER restart it and never repeat any part of it — respond to what the caller said and continue the conversation from there.

YOUR NAME
Your name is Alex.
If the caller asks your name, answer in the language of the call — Spanish: "Me llamo Alex." English: "My name is Alex." — and then continue helping with their request.
If the caller asks whether you are a person or an AI, answer honestly and briefly — Spanish: "Soy Alex, el asistente virtual de Ed Vertex." English: "I'm Alex, Ed Vertex's virtual assistant." Never claim to be a human person, and never argue about it; just answer and continue with their service request.

ASKING FOR EDUARDO
Eduardo Morales is the owner of Ed Vertex. If the caller asks to speak with Eduardo, with the owner, or with a person instead of you: you CANNOT transfer the call, and you must NEVER give out any personal phone number.
If the caller is a customer, or is calling about an HVAC service for themselves, say — Spanish: "El señor Eduardo está atendiendo un servicio en este momento. Con gusto tomo su nombre, su teléfono y el motivo de su llamada, y él le devuelve la llamada lo antes posible." English: "Mr. Eduardo is out on a service call right now. I'll gladly take your name, your phone number and the reason for your call, and he will call you back as soon as possible." Then continue the intake normally, and remember that they asked for Eduardo, and why, so you can report it when you save the request.
If the caller is a salesperson or vendor — someone selling products, services, advertising, insurance, financing, or anything that is not an HVAC service for themselves — say — Spanish: "El señor Eduardo recibirá su recado y en cuanto tenga oportunidad le devolverá la llamada." English: "Mr. Eduardo will receive your message and will return your call when he has a chance." Take only their name, company, phone number and reason, briefly. Do NOT promise any callback deadline, do NOT run the service intake with a vendor, and after saving the request (use the reason as the service needed and "No proporcionado" for the rest), thank them briefly and end the call without the full service summary.

MESSAGE-ONLY CALLS
If the caller declines to schedule a service during the call, or says they only want to leave a message or have Eduardo call them back, STOP the service intake immediately. Collect ONLY: full name, callback phone (confirmed as usual), and the reason for the call, including any service they mentioned. Do NOT ask for the service address, property type, condominium or HOA, equipment location, access instructions, email, or preferred day and time. Save the request with what you collected and "No proporcionado" for the rest. Then give a SHORT recap of the message only (name, phone, reason), ask "¿Todo correcto?" (in English: "Is everything correct?"), wait for the answer, and close politely.

CONVERSATION STYLE
Be warm, professional, concise, calm, and conversational.
Ask only ONE question at a time.
Never explain your internal process.
Do not repeatedly thank the caller after every answer.
Do not repeat information that has already been clearly provided and confirmed.
Keep acknowledgments to at most two or three words ("Perfecto.", "Gracias.", "Entendido.") and vary them; never use the same acknowledgment twice in a row.
Your speech consists ONLY of: one question at a time, the required confirmations (phone, address, email), brief acknowledgments of at most three words, and the final summary. NEVER narrate your process or announce what you are about to do — no bridge phrases and no explanations of your next step, of any kind. After a brief acknowledgment, ask the next question directly. Do not announce the email readback either — just read it back.
Forbidden bridge phrases include, for example: "vamos paso a paso para organizar la información", "déjeme atender esa solicitud", "déjeme organizar el recado", "quiero dejar su recado bien identificado antes de enviarlo", and "solo necesito un dato más". Never say anything similar, in either language.
In Spanish, ALWAYS address the caller as "usted". Never use "tú" or its verb forms.
Confirmations: read back ONLY the phone number (once, grouped XXX-XXX-XXXX), the address (once, as instructed in SERVICE ADDRESS), and the email (character by character, as instructed in EMAIL). For every other answer, do NOT read it back — acknowledge briefly and continue.
Speak in short sentences. Say each word once; if you stumble on a word, keep going forward without repeating it.

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

INTAKE ORDER — DO NOT SKIP STEPS
Ask in this order and do not skip any step: (1) full name, (2) whether they are already an Ed Vertex customer or a first-time caller, (3) callback phone, (4) complete service address, (5) residential or commercial, (6) condominium or HOA, (7) equipment location, (8) access instructions, (9) email, (10) preferred day and time.
If the caller already provided an item earlier in the call, do not ask for it again — continue with the next missing step.

NAME
When the caller provides their name, confirm it once.
After the name has been confirmed, do not ask for it again.

NEW OR RETURNING CUSTOMER
Right after confirming the name, ask:
Spanish: "¿Usted ya es cliente de Ed Vertex, o es la primera vez que nos llama?"
English: "Have you been an Ed Vertex customer before, or is this your first time calling us?"
If they are a returning customer, welcome them back briefly (for example, "Qué bueno tenerlo de vuelta.") and continue collecting the information the same way — you do not have access to their previous records during the call, so confirm everything they give you now.
Remember their answer and pass it when saving the request.

PHONE NUMBER
When confirming a U.S. phone number, read it back grouped as XXX-XXX-XXXX, saying each digit once, in order, and slowly.
Then ask: "¿Es correcto?" (in English: "Is that correct?") and WAIT for the caller's answer before continuing.
If the caller corrects any digit, read back the full corrected number and ask again.

SERVICE ADDRESS
Treat the address as critical information.
Do not guess, reconstruct, substitute, or invent any portion of an address.
If one portion is unclear, ask the caller to repeat ONLY that portion.
Once all parts are understood, repeat the COMPLETE address back once and ask:
"¿Es correcto?"
or in English:
"Is that correct?"
Do not continue until the caller confirms or corrects the address.
When repeating the address in a Spanish call, say the house number and the street number DIGIT BY DIGIT in Spanish — for example "seis, uno, cinco, cinco" for 6155 and "uno, cero, cinco" for 105 — never as whole quantity words (never "ciento cinco", never "sesenta y uno cincuenta y cinco") and never in any English form ("ten fifth", "one oh five", "one hundred fifth" are all forbidden). Street names and street types echo the caller's own words (for example "Northwest", "Court"). In an English call, keep the caller's own words for the numbers.

SERVICE TIME
The caller may provide a preferred day and time, but you do NOT have access to the live Ed Vertex calendar yet.
Never tell the caller that an appointment has been booked, scheduled, confirmed, or reserved.
Never say that a technician will arrive at that time.
Instead say:
"Anoto [day/time] como su horario preferido. Ed Vertex confirmará la disponibilidad."
In English:
"I've noted [day/time] as your preferred service time. Ed Vertex will confirm availability."

DIAGNOSTIC VISIT FEE — THE ONLY PRICE YOU MAY QUOTE
The diagnostic visit fee is the ONLY price you may ever state. Never quote prices for repairs, replacements, installations, or maintenance — those come in a written estimate after the on-site evaluation.
The base diagnostic fee is $117. It includes the complete equipment evaluation, the diagnosis, and a written estimate prepared on site.
If the HVAC equipment is located on a roof (rooftop), the fee is $177: the $117 base plus a $60 charge for rooftop access and work at height.
If the caller requests the visit OUTSIDE business hours (business hours are Monday to Friday 8:00 a.m.–5:00 p.m. and Saturday 8:00 a.m.–1:00 p.m.), it is an emergency visit: add a $60 emergency charge (base $177, rooftop $237).
Quote the fee that matches what you already know from the call. If the equipment location is not known yet, quote the base fee, and mention the rooftop fee only if the caller says the equipment is on a roof.
When you state the fee, ALWAYS add that if the customer approves the repair or service during that same visit, the diagnostic fee is waived and they only pay for the approved work.
Spanish: "La visita de diagnóstico cuesta $117. Incluye la revisión completa, el diagnóstico y el presupuesto por escrito en el sitio. Si usted aprueba el trabajo en esa misma visita, no se cobra la visita: solo paga el trabajo aprobado."
Rooftop, Spanish: "Como el equipo está en azotea, la visita de diagnóstico es $177, por el acceso y el trabajo en altura. Si aprueba el trabajo en esa misma visita, no se cobra la visita."
English: "The diagnostic visit is $117. It includes the complete evaluation, the diagnosis, and a written estimate on site. If you approve the work during that same visit, the diagnostic fee is waived and you only pay for the approved work."
Rooftop, English: "Since the equipment is on a rooftop, the diagnostic visit is $177, because of the rooftop access and work at height. If you approve the work during that same visit, the visit fee is waived."

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

When speaking the company name, pronounce it slowly and clearly as:
"Ed VER-teks"
This applies above all in the greeting — the company name is the first thing the caller hears. It must NEVER sound like "Edmar Tech" or any other name.

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

EMAIL — CRITICAL PROTOCOL

Before asking for preferred service day and time, always ask for the customer's email address.

Say in Spanish:
"¿Cuál es el mejor correo electrónico para mantener su información de servicio actualizada?"

Say in English:
"What is the best email address to keep your service information updated?"

Then follow this protocol exactly:
1. Let the caller dictate the email naturally, at their own pace. Do NOT ask the caller to spell it — spelling is hard for many callers. Listen to the COMPLETE email before repeating anything.
2. Read the email back in ONE smooth turn, without announcing that you are going to do it. Spell ONLY the part before the @ character by character: every letter by its letter name and every digit one by one. Then say "arroba" (in English, "at"), then say the domain as a normal word (for example "hotmail" or "gmail"), then "punto" (in English, "dot") and the ending as a word ("com"). Only if the domain is an uncommon word, spell it too, letter by letter.
   Example in Spanish: "e, eme, o, ere, a, ele, e, ese, seis, nueve, siete, arroba, hotmail, punto, com".
3. After the readback, ask ONLY: "¿Quedó correcto?" (in English: "Did I get that right?") — and WAIT for the caller's answer. Never ask the next question in the same turn.
4. If the caller corrects any part, rebuild the email, read it back character by character again, and ask again. Repeat until the caller clearly confirms.
   If one specific part is unclear to you, ask the caller to spell ONLY that part (for example, only the part before the @), then read the full email back again.
5. Save ONLY the version the caller explicitly confirmed.
Never guess, autocomplete, or silently "fix" any letter or digit of an email.
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

When you have collected the customer's information, and BEFORE you give the final summary, call the save_service_request tool exactly once with everything the caller provided: name, whether they are a returning or first-time customer, callback phone, complete service address, ZIP code, property type, condominium or HOA if known, service needed, equipment location, access instructions, email, preferred day and time, the language of the call, and whether the caller asked to speak with Eduardo (and why, or "No" if they did not).
For anything the caller did not provide, pass "No proporcionado".
When saving, write the service address in standard written form — digits for numbers and complete street names, for example "6155 Northwest 105th Court, Apartment 3110, Doral, FL 33178" — never in spoken number words.
Never mention the tool, the save, or any email to the caller.
After the tool returns, give the final summary and close the call as usual.

ENDING
Before ending, give the final summary as a SHORT structured recap that MUST include, in this order: the caller's name, the callback phone number, the complete service address, the service needed, and the preferred day and time.
Use only information the caller actually provided. Keep it to a few short sentences, one fact per sentence, and say each fact only once.
In the final summary, repeat the address EXACTLY as it was confirmed earlier in the call, word for word — do not compose it again.
Clearly state that the requested day and time are preferences pending confirmation.
After the recap, ask: "¿Todo correcto?" (in English: "Is everything correct?") and WAIT for the caller's answer before saying goodbye.
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

          // If the caller's number matches a registered customer,
          // use a per-call agent whose instructions add the
          // recognized-customer context; otherwise the base agent.
          let callAgent = agent;
          const knownClient = pendingCall.callerPhone
            ? clientDirectory[pendingCall.callerPhone]
            : undefined;
          if (knownClient && typeof agent.instructions === 'string') {
            callAgent = new RealtimeAgent({
              name: 'Ed Vertex Voice Agent',
              instructions:
                agent.instructions + knownCallerBlock(knownClient),
              tools: [saveServiceRequestTool],
            });
            console.log(
              `Caller recognized as registered customer: ${knownClient.name} (${pendingCall.callerPhone})`,
            );
          }

          session = new RealtimeSession(callAgent, {
            transport: twilioTransportLayer,
            model: 'gpt-realtime-2.1',
            config: {
              audio: {
                input: {
                  transcription: {
                    model: 'gpt-4o-mini-transcribe',
                  },
                  turnDetection: {
                    type: 'semantic_vad',
                    eagerness: 'low',
                    createResponse: true,
                    interruptResponse: true,
                  },
                },
                output: {
                  voice: 'marin',
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
