Subject: Residual personal information in lmsys/chatbot_arena_conversations — aggregate findings

Dear Chatbot Arena maintainers,

As part of a class research project (with instructor approval) on de-identification in public
LLM chat-log datasets, we measured residual personal information in a random sample of
5,000 English conversations from lmsys/chatbot_arena_conversations (revision 1b6335d42a1d).

Using Presidio (v2.2.364) plus rule-based secret detection, and after excluding your existing
redaction placeholders and obvious dummy values, we estimate (design-weighted, 95% CI):

- any quasi-identifier: 23.4% (22.2%–24.5%) of conversations
- any direct identifier: 14.8% (13.9%–15.8%) of conversations
- name: 14.8% (13.8%–15.7%) of conversations
- location: 11.9% (11.0%–12.7%) of conversations
- date: 11.3% (10.5%–12.2%) of conversations
- nrp: 5.2% (4.6%–5.8%) of conversations
- url: 2.0% (1.6%–2.4%) of conversations
- account id: 0.0% (0.0%–0.1%) of conversations
- ip address: 0.0% (0.0%–0.1%) of conversations
- email: 0.0% (0.0%–0.1%) of conversations

These are automated detector counts and include false positives; they have not yet been validated
against human labels. We have not attempted to identify any individual, linked any data to
external sources, or tested any credential.

We are not attaching examples. If useful, we can share the list of affected conversation IDs through
a private channel of your choosing so you can review them.

Best regards,
[Name, affiliation, contact]
