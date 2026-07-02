# Section 14.6: Azure Speech Services

> **Source:** Prosise, Ch. 14 - Speech-to-text and text-to-speech  
> **Prerequisites:** [Section 14.2](./section-02-azure-setup-and-authentication.md)  
> **Glossary:** [speech-to-text](../../GLOSSARY.md#speech-to-text) | [text-to-speech](../../GLOSSARY.md#text-to-speech) | [Azure Speech](../../GLOSSARY.md#azure-speech)  
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Voice-First Contoso Travel

Prosise's Contoso Travel assistant starts with **speech**: user asks a travel question aloud; **Azure Speech** transcribes to text, processes through Language services, and **synthesizes** the spoken response.

> **In plain English:** Microphone in → text out (STT). Answer text in → speaker out (TTS). The user never types if they don't want to.

---

## Speech-to-Text (STT)

```python
# pip install azure-cognitiveservices-speech
import os
import azure.cognitiveservices.speech as speechsdk

speech_config = speechsdk.SpeechConfig(
    subscription=os.environ['AZURE_SPEECH_KEY'],
    region=os.environ['AZURE_SPEECH_REGION'],
)
speech_config.speech_recognition_language = "en-US"

audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config, audio_config=audio_config
)

print("Speak your travel question...")
result = recognizer.recognize_once()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("Recognized:", result.text)
elif result.reason == speechsdk.ResultReason.NoMatch:
    print("No speech recognized")
elif result.reason == speechsdk.ResultReason.Canceled:
    print("Canceled:", result.cancellation_details.reason)
```

See [speech-to-text in glossary](../../GLOSSARY.md#speech-to-text).

---

## Continuous Recognition

For longer utterances:

```python
done = False

def handle_result(evt):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Partial/final:", evt.result.text)

def stop_cb(evt):
    global done
    done = True

recognizer.recognized.connect(handle_result)
recognizer.session_stopped.connect(stop_cb)
recognizer.start_continuous_recognition()
while not done:
    time.sleep(0.5)
recognizer.stop_continuous_recognition()
```

Contoso CLI can use single-shot; mobile apps prefer continuous.

---

## Text-to-Speech (TTS)

**Neural voices** sound natural - multiple locales and personas:

```python
speech_config = speechsdk.SpeechConfig(
    subscription=os.environ['AZURE_SPEECH_KEY'],
    region=os.environ['AZURE_SPEECH_REGION'],
)
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

text = "Your flight to Paris is confirmed for July 15th."
result = synthesizer.speak_text_async(text).get()

if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech played successfully")
```

Save to file:

```python
audio_config = speechsdk.audio.AudioOutputConfig(filename="response.wav")
synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config, audio_config=audio_config
)
synthesizer.speak_text_async(text).get()
```

See [text-to-speech in glossary](../../GLOSSARY.md#text-to-speech).

---

## SSML for Prosody Control

**Speech Synthesis Markup Language** controls emphasis, pauses, pronunciation:

```xml
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="en-US-JennyNeural">
    Welcome to <emphasis level="strong">Contoso Travel</emphasis>.
    <break time="500ms"/>
    Your hotel near the <sub alias="Louvre Museum">Louvre</sub> is booked.
  </voice>
</speak>
```

```python
synthesizer.speak_ssml_async(ssml_string).get()
```

Use SSML when brand names mispronounced by default voices.

---

## Language Support

Match STT language to user locale:

```python
speech_config.speech_recognition_language = "fr-FR"  # French input
speech_config.speech_synthesis_voice_name = "fr-FR-DeniseNeural"
```

List voices: `SpeechSynthesizer.get_voices_async()` or Azure portal Voice Gallery.

See [Azure Speech in glossary](../../GLOSSARY.md#azure-speech).

---

## Transcribe Audio Files

```python
audio_config = speechsdk.audio.AudioConfig(filename="question.wav")
recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config, audio_config=audio_config
)
result = recognizer.recognize_once()
```

Batch transcription API for long recordings (meetings, call centers) - async job pattern.

---

## Privacy and Tradeoffs

| Benefit | Risk |
|---------|------|
| No custom acoustic model training | Audio sent to cloud |
| Excellent multilingual STT | Network latency |
| Neural TTS quality | Per-minute billing |

Air-gapped environments need on-prem speech (Whisper self-hosted) - build path from Course 3.

---

## Contoso Speech Loop

```
1. STT: user question → text
2. Language + Translator (if needed)
3. Business logic + retrieval
4. Language: sentiment-aware response text
5. TTS: speak response
```

Log **which services fired** per request ([Lab 14](./section-lab-14-contoso-travel-assistant.md)).

---

## Key Takeaways

1. **Azure Speech** provides STT and neural TTS via SDK
2. **Single-shot** vs **continuous** recognition for short vs long input
3. **SSML** fine-tunes pronunciation and pacing
4. Match **voice locale** to user language
5. Consider **privacy/latency** when sending audio to cloud

---

## Check Your Understanding

1. What SDK class performs speech recognition?
2. How enable French input recognition?
3. What is SSML used for?
4. Name one privacy tradeoff of cloud STT.
5. Where does TTS sit in Contoso pipeline?

---

## Latency Budget

For voice assistants, budget end-to-end latency:

$$
L_{\text{total}} \approx L_{\text{STT}} + L_{\text{NLP}} + L_{\text{TTS}} + L_{\text{network}}
$$
> **Readable form:** total latency ≈ speech-to-text + language processing + text-to-speech + network round trips

Chain services in-region to minimize $L_{\text{network}}$ ([Section 14.2](./section-02-azure-setup-and-authentication.md)).

---

## References

- Prosise, Ch. 14 - Speech
- Speech SDK docs - [https://learn.microsoft.com/azure/ai-services/speech-service/quickstarts/setup-platform](https://learn.microsoft.com/azure/ai-services/speech-service/quickstarts/setup-platform)
- [GLOSSARY.md](../../GLOSSARY.md) | [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

**Next:** [Section 14.7 - Contoso Travel: Multi-Service App](./section-07-contoso-travel-multi-service-app.md)



