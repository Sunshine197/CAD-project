# Dr7 API integration notes

Research checked 19 November 2026 against Dr7's official documentation and API-access pages.

Dr7's API documentation describes a Bearer API key and `POST https://dr7.ai/api/v1/medical/chat/completions`. Its example request uses `medgemma-4b-it`, a `messages` array, `max_tokens`, and `temperature`. The response example uses `choices[0].message.content`. The same page lists HTTP 401, 402, 429, and 500 error cases. The API-access page describes MedGemma text generation and MedSigLIP image/text capabilities. [API documentation](https://dr7.ai/docs) [API access](https://dr7.ai/access-api)

The docs include a medical disclaimer stating that the information is not medical advice, that MedGemma is for research and development rather than actual patient care without validation and regulatory approval, and that clinical applications require thorough validation and expert oversight. [API documentation, disclaimer section](https://dr7.ai/docs)

The docs do not establish that generated reports are hallucination-free, clinically validated for this project's use, or suitable for autonomous diagnosis. They also do not explain in the retrieved pages how API prompts are retained, used for training, or governed by a healthcare data agreement. The linked privacy and terms pages could not be retrieved during this research. Therefore, this project does not transmit images, patient identifiers, or other direct identifiers to Dr7. The app requires an explicit user action and presents the response only as an unverified draft. Users must independently review Dr7's current policies/contract and avoid sensitive or identifiable information unless appropriate approvals and safeguards are in place.

## Implementation decision

The deterministic, local report is the default. The optional Dr7 call sends the full classifier result (all class scores, score semantics, calibration status, model/checkpoint/preprocessing metadata, limitations) and non-identifying examination fields as text. It does not send the image. The LLM response is neither automatically fact-checked nor clinically validated; a prompt cannot guarantee hallucination-free output. Every LLM draft is labeled unverified and requires clinician review.
