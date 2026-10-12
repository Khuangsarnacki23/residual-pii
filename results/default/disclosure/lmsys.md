Subject: Residual personal information in lmsys/lmsys-chat-1m — aggregate findings

Dear LMSYS-Chat-1M maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
5,000 English conversations from lmsys/lmsys-chat-1m (revision 200748d9d3cd).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- any quasi-identifier: 35.2% (34.0%–36.5%) of conversations
- date: 21.0% (19.9%–22.2%) of conversations
- location: 16.6% (15.6%–17.7%) of conversations
- any direct identifier: 12.1% (11.1%–12.9%) of conversations
- name: 12.0% (11.1%–12.8%) of conversations
- nrp: 8.7% (8.0%–9.5%) of conversations
- url: 8.6% (7.8%–9.3%) of conversations
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
