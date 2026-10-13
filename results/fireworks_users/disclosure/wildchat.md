Subject: Residual personal information in allenai/WildChat-1M — aggregate findings

Dear WildChat-1M maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
6,765 English conversations from allenai/WildChat-1M (revision 7d6490e46228).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- any quasi-identifier: 64.8% (62.1%–67.5%) of conversations
- any direct identifier: 64.3% (61.9%–66.5%) of conversations
- name: 64.2% (61.7%–66.4%) of conversations
- date: 35.5% (32.4%–38.7%) of conversations
- nrp: 28.8% (27.0%–30.8%) of conversations
- location: 27.4% (24.8%–30.2%) of conversations
- url: 10.0% (8.3%–12.0%) of conversations
- account id: 0.2% (0.1%–0.3%) of conversations
- email: 0.1% (0.0%–0.3%) of conversations
- ip address: 0.1% (0.1%–0.2%) of conversations
- api key token: 0.0% (0.0%–0.1%) of conversations
- phone: 0.0% (0.0%–0.1%) of conversations

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
