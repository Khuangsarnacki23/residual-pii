Subject: Residual personal information in lmsys/lmsys-chat-1m — aggregate findings

Dear LMSYS-Chat-1M maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
4,671 English conversations from lmsys/lmsys-chat-1m (revision 200748d9d3cd).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- any quasi-identifier: 36.4% (35.0%–37.8%) of conversations
- date: 21.5% (20.4%–22.6%) of conversations
- location: 17.2% (16.1%–18.2%) of conversations
- any direct identifier: 12.4% (11.5%–13.3%) of conversations
- name: 12.3% (11.4%–13.3%) of conversations
- nrp: 9.0% (8.2%–9.9%) of conversations
- url: 4.0% (3.5%–4.6%) of conversations
- account id: 0.0% (0.0%–0.1%) of conversations
- ip address: 0.0% (0.0%–0.1%) of conversations
- api key token: 0.0% (0.0%–0.1%) of conversations

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
