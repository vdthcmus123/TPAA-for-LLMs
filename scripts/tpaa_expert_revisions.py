"""Expert responses rewritten as statements (not questions).

Real MRBench Expert turns are questions only ~68% (MathDial) / ~53% (Bridge)
of the time; these overrides bring the TPAA set closer to that mix and add
label variety (some partly reveal the fix or the answer).
Keys: ("MD", n) or ("BR", n), 1-based item numbers.
"""

EXPERT_OVERRIDE = {
    ("MD", 2): "If you only had 3 buses, 25 people would have to stay behind at school.",
    ("MD", 5): "Wednesday is 7 fewer than Tuesday, not Monday.",
    ("MD", 8): "Remember the shop still paid for the 6 shirts it didn't sell.",
    ("MD", 11): "The 6 litres is for every 100 km, so first work out how many hundreds of km are in 450.",
    ("MD", 13): "Good start, but remember he is giving the walls two coats.",
    ("MD", 18): "Diego already has $50, so he only needs to save the rest.",
    ("MD", 24): "That's only one week of eggs, he sells them for 4 weeks.",
    ("MD", 32): "There are 2 adults, so you need to take away two adult tickets.",
    ("MD", 44): "Remember he spends $3 on snacks before he saves anything.",
    ("MD", 62): "One box isn't enough because 24 kg is more than the 20 kg a box can hold.",
    ("MD", 78): "Any part of an hour is charged as a full hour, so the 25 minutes costs $2 as well.",
    ("MD", 82): "Only the 6 cubic metres above 10 are charged at $3.",
    ("MD", 88): "She drives to work and back each day, so that's 36 miles a day.",
    ("MD", 91): "You're comparing a week's pay with a month's pay - Job A needs to be multiplied by 4 first.",
    ("MD", 99): "Twice as large means 2 x 2.5, not 2 + 2.5.",
    ("BR", 2): "When we round to the nearest hundred we look at the tens digit, and here it's a 5.",
    ("BR", 4): "Some apples went into the basket, so there should be fewer left on the table.",
    ("BR", 7): "That's the area of the whole rectangle. A triangle is half of that.",
    ("BR", 9): "Remember 1 kilometer is 1,000 meters.",
    ("BR", 13): "Great thinking, but when the door opens wider the angle gets bigger than a right angle.",
    ("BR", 14): "Remember we multiply before we subtract.",
    ("BR", 17): "Edges are where two faces meet - let's count them together on the cube.",
    ("BR", 21): "Speed is distance divided by time, so we divide 180 by 2.5.",
    ("BR", 25): "Look at the 6 in the hundredths place - it's more than 5.",
    ("BR", 26): "The pencil starts at 1, not 0, so we need to count the spaces between 1 and 7.",
    ("BR", 29): "Quad means four, like a quad bike has four wheels.",
    ("BR", 32): "Careful, 40 is 4 tens, not 4 hundreds.",
    ("BR", 35): "12 is the total number of votes. We need the difference between 7 and 5.",
    ("BR", 37): "Let's change the 3 feet into inches first. Each foot is 12 inches.",
    ("BR", 40): "The 7 is in the thousands place, so it's worth a lot more than 7.",
    ("BR", 45): "Parentheses first! Work out 8 - 3 before you multiply.",
    ("BR", 47): "Even when 6 goes into 0 zero times, we still write a 0 in the answer.",
    ("BR", 50): "It goes 60 miles every hour, so in 2.5 hours it should go more than 60 miles.",
}
