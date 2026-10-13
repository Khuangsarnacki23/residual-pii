Subject: Residual personal information in allenai/WildChat-1M — aggregate findings

Dear WildChat-1M maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
19,517 English conversations from allenai/WildChat-1M (revision 7d6490e46228).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- any quasi-identifier: 62.9% (61.5%–64.2%) of conversations
- any direct identifier: 60.3% (58.9%–61.6%) of conversations
- name: 60.2% (58.8%–61.5%) of conversations
- date: 32.3% (30.7%–33.8%) of conversations
- nrp: 32.3% (31.1%–33.5%) of conversations
- location: 26.3% (24.9%–27.9%) of conversations
- url: 11.1% (10.1%–12.1%) of conversations
- account id: 0.4% (0.3%–0.5%) of conversations
- email: 0.2% (0.1%–0.2%) of conversations
- ip address: 0.2% (0.1%–0.2%) of conversations
- api key token: 0.1% (0.1%–0.1%) of conversations
- phone: 0.0% (0.0%–0.0%) of conversations

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
