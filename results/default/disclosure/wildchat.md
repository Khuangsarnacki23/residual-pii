Subject: Residual personal information in allenai/WildChat-1M — aggregate findings

Dear WildChat-1M maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
5,000 English conversations from allenai/WildChat-1M (revision 7d6490e46228).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- any quasi-identifier: 53.8% (52.5%–55.1%) of conversations
- any direct identifier: 44.5% (43.2%–45.9%) of conversations
- name: 44.4% (43.0%–45.7%) of conversations
- nrp: 33.1% (31.9%–34.3%) of conversations
- date: 20.5% (19.4%–21.5%) of conversations
- location: 18.5% (17.6%–19.6%) of conversations
- url: 9.8% (9.0%–10.6%) of conversations
- api key token: 0.2% (0.1%–0.4%) of conversations
- account id: 0.2% (0.1%–0.4%) of conversations
- ip address: 0.2% (0.1%–0.3%) of conversations
- email: 0.1% (0.0%–0.2%) of conversations
- phone: 0.0% (0.0%–0.1%) of conversations

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
