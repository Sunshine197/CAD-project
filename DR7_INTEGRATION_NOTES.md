# Dr7 API integration and privacy notes

**Research checked:** 19 November 2026, against Dr7's published API documentation, API-access FAQ, privacy policy, and terms of service. This is a project risk review, not legal advice or independent verification of vendor claims.

## API and medical-use limits

Dr7's API documentation describes a Bearer API key and `POST https://dr7.ai/api/v1/medical/chat/completions`. It shows model `medgemma-4b-it`, a `messages` array, and an OpenAI-compatible response. The API-access page says the free plan is web-chat-only; API usage requires a paid plan or credits. [API documentation](https://dr7.ai/docs) [API access](https://dr7.ai/access-api)

The published terms describe MedGemma as a research and educational platform. They say responses may be inaccurate, incomplete, or outdated, disclaim medical accuracy and warranties, and prohibit clinical diagnosis, treatment planning, patient-care decisions, automated medical decision-making, and processing real patient data. The terms were marked “Last updated: October 2026.” [Terms of Service](https://dr7.ai/terms-of-service)

The API-access FAQ says not to send directly identifiable patient information unless there is a signed data-processing agreement, and says to remove names, ID numbers, and other identifiers. It describes responses as informational and for research. That FAQ does not clearly reconcile with the broader terms' prohibition on processing real patient data, nor does it define whether de-identified real-patient-derived model outputs are allowed. A signed agreement cannot be assumed to override those restrictions unless Dr7 confirms the scope in writing. [API access](https://dr7.ai/access-api) [Terms of Service](https://dr7.ai/terms-of-service)

## Published privacy statements and unresolved points

The privacy policy says the service processes medical text and images for demonstration and research. It says inputs are used for inference, performance analysis, and platform improvement; says query/response content is purged after processing and within 24 hours; and says anonymized technical logs may be retained for 90 days. It also asserts use of Google Cloud, encryption, access controls, and healthcare compliance, including HIPAA, SOC 2 Type II, and ISO 27001. The page is marked “Last updated: December 2024.” These are vendor-published statements; this review did not independently verify certifications, infrastructure, deletion behavior, or whether every statement applies to API traffic. No API-specific retention schedule, public DPA text, or independent compliance evidence was located in the pages reviewed. [Privacy Policy](https://dr7.ai/privacy-policy)

The stated use of inputs for “performance analysis” and “platform improvement” means the public policy does not amount to a clearly documented no-training/no-secondary-use guarantee. The text also does not give enough detail to establish specific subprocessors, data location for each API request, incident response terms, or contractual rights for this project's data.

## Project decision

**Do not send real-patient data to Dr7 or use Dr7 output for clinical decisions under the currently published terms.** This includes not assuming that omitting images, names, or dates makes a score payload from a real patient permissible. Before any real-patient-derived data is considered, obtain Dr7's written clarification of the conflict and a signed agreement that explicitly authorizes the precise research use and addresses retention/deletion, training or service improvement, subprocessors, data location, security, and permitted medical use. Also confirm the project's own data-use and institutional approvals. Until then, any live API experiment should use synthetic, non-patient test data only.

The app now presents the restrictions before the optional Dr7 form and requires the user to affirm that the request contains no real-patient data or direct identifiers, is for research/education only, and will not be used for clinical decisions. This is an attestation, not a vendor authorization mechanism or a substitute for checking data provenance. The local deterministic report remains the default. The optional client still sends text only, but the full model scores and metadata can themselves be health-related information. No live Dr7 request was made during this review.

## Existing technical boundaries

The optional client sends the full classifier result, including all class scores, score semantics, calibration status, model/checkpoint/preprocessing metadata, and limitations, plus non-identifying examination fields. It does not send the image, original filename, patient name, or date of birth. The response is not automatically fact-checked or clinically validated, and a prompt cannot guarantee a hallucination-free report. The output remains an unverified draft requiring clinician review.
