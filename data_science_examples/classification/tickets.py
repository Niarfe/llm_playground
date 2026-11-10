"""
Forty support tickets, labelled by hand, and the four labels they fall into.

WHY HAND-WRITTEN AND HAND-LABELLED
    Asking a model to generate labelled training data is the obvious
    shortcut and it quietly destroys the experiment. The model under test
    here is an LLM; if an LLM also wrote and labelled the data, then its
    accuracy partly measures how well it agrees with itself. Generator and
    subject share a prior, so the eval confirms the prior instead of testing
    it. The score comes out high and means nothing.

    Forty rows typed by a person is a boring hour and an honest number.

THE SPLIT
    Eight tickets are marked `split="train"`. Those are the only ones the
    few-shot prompt is allowed to show the model. The other thirty-two are
    the test set.

    This is not ceremony. Put a test item in the prompt and the model can
    answer it from the prompt, the score goes up, and you have measured
    nothing but your own leak. It is the most common way a promising result
    turns out to be fake, and it is one line of code to get wrong.

AMBIGUOUS BY DESIGN
    Five tickets carry `ambiguous=True`. Each one has a real claim on two
    labels, and the label recorded here is a judgement, not a fact.

        #3   card declined but the account reads as current
             -> billing, or a bug in the status display?
        #35  an owner who cannot see the billing section
             -> a permissions problem, or a bug?
        #37  SSO login looping back to the sign-in page
             -> access, or a bug in the SSO handler?
        #38  an account suspended for no stated reason
             -> access, or non-payment, which is billing?
        #19  scheduled emails sent twice
             -> a bug, unless the schedule was configured twice

    Two human annotators would disagree on these. That means the achievable
    accuracy on this set is not 100%, and a model that "fails" #35 may be
    more right than the label is. run.py reports the score both with and
    without these five, and the gap between those two numbers is the part
    worth looking at.

    Every benchmark you will ever read has this problem and almost none of
    them tell you the size of it.
"""

LABELS = ["billing", "bug", "feature_request", "account_access"]

LABEL_NOTES = {
    "billing":        "charges, invoices, refunds, prices, payment methods",
    "bug":            "the product behaves differently from how it should",
    "feature_request":"something the product does not do yet",
    "account_access": "logins, passwords, permissions, membership, ownership",
}

TICKETS = [
    # -- billing -----------------------------------------------------------
    dict(id=1,  text="I was charged twice for last month's subscription.",
         label="billing", split="train"),
    dict(id=2,  text="Can I get a copy of the invoice for September?",
         label="billing", split="train"),
    dict(id=3,  text="My card was declined but the service says my account is current.",
         label="billing", split="test", ambiguous=True),
    dict(id=4,  text="Why did my monthly price go up from 29 to 39?",
         label="billing", split="test"),
    dict(id=5,  text="Please cancel my plan and refund the last payment.",
         label="billing", split="test"),
    dict(id=6,  text="We need to switch from paying by credit card to being invoiced quarterly.",
         label="billing", split="test"),
    dict(id=7,  text="The receipt shows the wrong company name on it.",
         label="billing", split="test"),
    dict(id=8,  text="I upgraded mid-month, how is the proration calculated?",
         label="billing", split="test"),
    dict(id=9,  text="Our purchase order number needs to appear on future bills.",
         label="billing", split="test"),
    dict(id=10, text="You billed us in USD but our contract says EUR.",
         label="billing", split="test"),

    # -- bug ---------------------------------------------------------------
    dict(id=11, text="The export button does nothing when I click it.",
         label="bug", split="train"),
    dict(id=12, text="The dashboard shows -1 users after I removed someone from the team.",
         label="bug", split="train"),
    dict(id=13, text="Uploading a CSV over about 5MB fails with a blank error.",
         label="bug", split="test"),
    dict(id=14, text="Dates display as 1970 for rows that were created yesterday.",
         label="bug", split="test"),
    dict(id=15, text="The app logs me out every few minutes on Safari.",
         label="bug", split="test"),
    dict(id=16, text="Search returns results that do not contain the search term.",
         label="bug", split="test"),
    dict(id=17, text="Totals in the report do not add up to the sum of the rows.",
         label="bug", split="test"),
    dict(id=18, text="After the last update the sidebar is cut off on a 13 inch screen.",
         label="bug", split="test"),
    dict(id=19, text="Scheduled emails went out twice this morning.",
         label="bug", split="test", ambiguous=True),
    dict(id=20, text="The timezone on the activity log is off by four hours.",
         label="bug", split="test"),

    # -- feature_request ---------------------------------------------------
    dict(id=21, text="Please add a dark mode.",
         label="feature_request", split="train"),
    dict(id=22, text="It would help if we could tag tickets with more than one label.",
         label="feature_request", split="train"),
    dict(id=23, text="Any chance of a Slack integration?",
         label="feature_request", split="test"),
    dict(id=24, text="We would like to export to Parquet, not just CSV.",
         label="feature_request", split="test"),
    dict(id=25, text="Can the dashboard remember my filters between visits?",
         label="feature_request", split="test"),
    dict(id=26, text="Add the ability to schedule a report weekly.",
         label="feature_request", split="test"),
    dict(id=27, text="An API endpoint for bulk user creation would save us hours.",
         label="feature_request", split="test"),
    dict(id=28, text="Could you support single sign on with Okta?",
         label="feature_request", split="test"),
    dict(id=29, text="I want to see a chart of ticket volume over time.",
         label="feature_request", split="test"),
    dict(id=30, text="Please let us rename the default project.",
         label="feature_request", split="test"),

    # -- account_access ----------------------------------------------------
    dict(id=31, text="I forgot my password and the reset email never arrives.",
         label="account_access", split="train"),
    dict(id=32, text="Locked out after too many failed login attempts.",
         label="account_access", split="train"),
    dict(id=33, text="My two factor device was replaced, how do I re-enroll?",
         label="account_access", split="test"),
    dict(id=34, text="Please remove admin rights from a teammate who left.",
         label="account_access", split="test"),
    dict(id=35, text="I cannot see the billing section even though I am an owner.",
         label="account_access", split="test", ambiguous=True),
    dict(id=36, text="A new hire needs to be added to our workspace.",
         label="account_access", split="test"),
    dict(id=37, text="The SSO login loops back to the sign in page.",
         label="account_access", split="test", ambiguous=True),
    dict(id=38, text="My account says it is suspended and I do not know why.",
         label="account_access", split="test", ambiguous=True),
    dict(id=39, text="Need to transfer ownership of the workspace to another person.",
         label="account_access", split="test"),
    dict(id=40, text="The invitation link expired before my colleague could use it.",
         label="account_access", split="test"),
]

TRAIN = [t for t in TICKETS if t["split"] == "train"]
TEST  = [t for t in TICKETS if t["split"] == "test"]
