def prompt_few_shot(context, newcomer, reply):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a model that identifies whether a pair of comments on Reddit are using counterspeech, hate speech, or other speech. You provide your answers in the format (newcomer: prediction, reply: prediction).<|eot_id|><|start_header_id|>user<|end_header_id|>

    Hate speech is defined by the United Nations as “any kind of communication in speech, writing or behaviour, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.” Counterspeech is speech that opposes hate speech. ‘Other’ speech is neither counterspeech nor hate speech.

    The comments come from a hate subreddit, so speech that calls out the community for being hateful or ideologies of hate may qualify as counterspeech. Note that counterspeech may still be hostile or contain insults, but as long as it is speaking out against harmful speech directed at identity groups, such as racism, misogyny, anti-LGBTQ speech, or ableism, it will still qualify as counterspeech.

    You can assign one of three labels to each comment. Note that the comments may each have a different label:
    “Other”
    “Hate speech”
    “Counterspeech”

    You will be given three comments: “context,” “newcomer,” and “reply.” The “newcomer” comment is a reply to the “context” comment, and the “reply” is a reply to the “newcomer” comment. Your goal is to identify whether the “newcomer” and “reply” comments are neutral, hate speech, or counterspeech. Please provide your answer in the format of (newcomer: prediction, reply: prediction). Do not include any additional details in your reply.

    Context: {{Check out this rune reading faggot}}
    Newcomer: {{There are more rune reading faggots in China than Latin descendant faggots in America}}
    Reply: {{Ok buster, remember the opium wars.}}
    Answer: (Newcomer: hate speech, reply: other)

    Context: {{I like all races except the blacks}}
    Newcomer: {{I'm sure no races like you}}
    Reply: {{If that's the best you can do, maybe you should just keep your comments to yourself.}}
    Answer: (Newcomer: counterspeech, reply: other)

    Context: {{That's what liberals want, that everybody becomes transgender, every single person in the planet should get a sex change according to them, so it will be easier for them to take our children}}
    Newcomer: {{This is the most illogical conclusion that I've ever read.}}
    Reply: {{Doesnt matter, you liberals will never win our kids. MAGA}}
    Answer: (Newcomer: counterspeech, reply: hate speech)

    Context: {{How do you go about getting them}}
    Newcomer: {{You know how to post on Reddit but not how to Google?}}
    Reply: {{I know how to Google.}}
    Answer: (Newcomer: other, reply: other)

    Context: {{I guess it's never a good idea to censor yourself and not express opinions that you think are legitimate; but on the other side - you wouldn't expect a TV network to keep relationships with a guy who has a twitter like that.}}
    Newcomer: {{Nice logic.  Unrelated question: When did you become such a pussy?  So tired of retards like you caving in on free speech.  His twitter is hilarious, it's meant to be funny.}}  
    Reply: {{wtf is wrong with u blockheads, where the fuck did i say that his twitter is not funny? and 'caving in on free speech'? Also where did I say that I agree with Adult Swim's decision. I srs dont understand what are u all talking about.}}
    Answer: (Newcomer: hate speech, reply: other)

    Context: {{{context}}}
    Newcomer: {{{newcomer}}}
    Reply: {{{reply}}}
    Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return prompt
