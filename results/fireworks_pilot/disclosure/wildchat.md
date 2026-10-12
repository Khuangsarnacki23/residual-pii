Subject: Residual personal information in allenai/WildChat-1M — aggregate findings

Dear WildChat-1M maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
426 English conversations from allenai/WildChat-1M (revision 7d6490e46228).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- name: 52.1% (47.9%–56.6%) of conversations
- any direct identifier: 52.1% (47.9%–56.6%) of conversations
- any quasi-identifier: 52.1% (47.4%–57.0%) of conversations
- location: 34.5% (30.0%–38.5%) of conversations
- date: 30.8% (26.8%–34.7%) of conversations
- nrp: 20.7% (17.4%–24.7%) of conversations
- url: 14.1% (11.0%–17.6%) of conversations

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
