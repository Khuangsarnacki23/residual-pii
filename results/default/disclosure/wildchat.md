Subject: Residual personal information in allenai/WildChat-1M — aggregate findings

Dear WildChat-1M maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
3,881 English conversations from allenai/WildChat-1M (revision 7d6490e46228).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- any quasi-identifier: 40.5% (39.1%–42.0%) of conversations
- any direct identifier: 29.1% (27.8%–30.5%) of conversations
- name: 28.8% (27.5%–30.2%) of conversations
- date: 24.7% (23.5%–26.0%) of conversations
- location: 23.1% (21.7%–24.3%) of conversations
- nrp: 14.6% (13.5%–15.7%) of conversations
- url: 12.6% (11.6%–13.8%) of conversations
- api key token: 0.3% (0.1%–0.5%) of conversations
- account id: 0.3% (0.1%–0.5%) of conversations
- ip address: 0.2% (0.1%–0.4%) of conversations
- email: 0.1% (0.0%–0.3%) of conversations
- phone: 0.0% (0.0%–0.1%) of conversations

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
