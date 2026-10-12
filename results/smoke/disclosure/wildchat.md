Subject: Residual personal information in allenai/WildChat-1M — aggregate findings

Dear WildChat-1M maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
100 English conversations from allenai/WildChat-1M (revision 7d6490e46228).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- any quasi-identifier: 43.8% (33.9%–52.8%) of conversations
- name: 29.9% (20.9%–38.9%) of conversations
- any direct identifier: 29.9% (20.9%–38.9%) of conversations
- location: 25.9% (19.0%–33.9%) of conversations
- date: 25.9% (17.8%–34.8%) of conversations
- nrp: 13.0% (7.9%–19.0%) of conversations
- url: 12.9% (7.0%–18.0%) of conversations
- ip address: 1.0% (0.0%–3.0%) of conversations

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
