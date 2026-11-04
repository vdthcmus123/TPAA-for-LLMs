"""Longer first-person walk-throughs (MathDial-style) and probe phrasing variants.

EXPLAIN[i] replaces the first student explanation (turn index 1) of item i.
PROBES rotates the generic opening probe so it is not the same sentence every time
(phrasings modelled on the variety seen in MathDial teacher turns).
"""

EXPLAIN = {
1: "Sure. First I worked out how many cookies Lena baked altogether. There are 4 trays and each tray has 15 cookies, so I multiplied 4 by 15 and got 60 cookies. Then the question says she packs them equally into 6 boxes, so I shared the 60 cookies equally between the 6 boxes. 60 divided by 6 is 10, so each box has 10 cookies in it.",
3: "Sure. The side of the field is 60 meters long and Omar puts a post every 5 meters. So I needed to find how many 5s fit into 60, because each 5 meters has a post. I divided 60 by 5 and got 12. That means there are 12 posts along the side of the field, so Omar uses 12 posts.",
4: "Sure. The jacket costs $120 and Mia has two things that make it cheaper, the $10 coupon and the 25% discount. I took off the $10 coupon first, which gave me $110. Then I worked out 25% of $110 by multiplying 110 by 0.25, which is $27.50. Then I subtracted $27.50 from $110 and got $82.50, so that is what Mia pays.",
5: "Sure. On Monday she reads 18 pages. Tuesday is twice as many as Monday, so I doubled 18 and got 36 pages. Wednesday is 7 fewer pages, so I took 7 away from 18 and got 11 pages. Then to get the total for the three days I added 18, 36 and 11 together, which gave me 65 pages altogether.",
7: "Sure. I said Ana's age is A and Carlos is A + 4 because he is 4 years older. The sum of their ages in 6 years will be 40, so to get the sum of their ages now I subtracted the 6 years and got 34. Then I took away the 4 extra years that Carlos has, which left 30, and that is two lots of Ana's age. So I split 30 into two equal parts and got 15, so Ana is 15.",
12: "Sure. First I worked out her average on the first three tests by adding 78, 85 and 92 and dividing by 3, which gave me 85. She wants an average of 88, so she is 3 points short of what she wants. So on the last test she needs to make up those 3 points, which means she needs 3 more than 88. I added 88 and 3 and got 91, so she needs 91 on the fourth test.",
13: "Sure. I started by finding the area of one wall. Each wall is 5 meters long and 3 meters high, so one wall is 5 times 3, which is 15 square meters. There are 4 walls, so I multiplied 15 by 4 and got 60 square meters for the whole room. Then one can covers 12 square meters, so I divided 60 by 12 and got 5 cans of paint.",
17: "Sure. The garden is 12 meters long and 8 meters wide, and the path is 1 meter wide all around it. So I made the garden bigger by the path: I added 1 to the length to get 13 and 1 to the width to get 9. The big rectangle is 13 times 9, which is 117 square meters. Then I took away the garden, which is 12 times 8, 96 square meters, to leave just the path, which is 21 square meters.",
23: "Sure. There are 32 students and 3/8 of them are boys, so first I found 3/8 of 32. I divided 32 by 8 to get 4 and then multiplied by 3 to get 12 boys. Then the question says half of them wear glasses, so I halved 12 and got 6. So 6 of them wear glasses, and that is my answer for the girls.",
27: "Sure. The store had 250 copies. On Saturday they sold 40% and on Sunday they sold 30%, so altogether they sold 40% plus 30%, which is 70% of the books. Then I found 70% of 250 by multiplying 250 by 0.70 and got 175 books sold. To find how many were left I took 175 away from 250 and got 75 copies left after Sunday.",
28: "Sure. Chickens have 2 legs each, so the 9 chickens have 9 times 2, which is 18 legs. Cows have 4 legs each, so the 6 cows have 6 times 4, which is 24 legs. Then I added the chicken legs and the cow legs together, 18 plus 24, and got 42 legs in the barn altogether.",
30: "Sure. First I found how many pages Zoe has to print. She needs 15 copies and each copy has 24 pages, so 15 times 24 is 360 pages. The printer prints 12 pages every minute, so to get the time I multiplied the 360 pages by the 12 pages per minute. That gave me 4320, so the printing takes 4320 minutes.",
33: "Sure. Liam is awake from 7 a.m. to 10 p.m., so I subtracted 7 from 10 to find how long he is awake, which is 3 hours. He drinks 250 milliliters every hour, so in 3 hours he drinks 3 times 250, which is 750 milliliters. The question wants liters, so I divided 750 by 1000 and got 0.75 liters.",
41: "Sure. The meal is $40. There is a 10% service charge and a 5% tax, so altogether that is 15% extra on top of the meal. I found 15% of $40 by multiplying 40 by 0.15, which is $6. Then I added the $6 to the $40 for the meal, so the final bill is $46.",
44: "Sure. Tom gets $12 every week and he puts 1/3 of his allowance into his savings jar. A third of $12 is $4, so he saves $4 every week. Then I multiplied the $4 by the 8 weeks, which is $32, so he saves $32 in 8 weeks.",
49: "Sure. Leo buys notebooks that cost $2.35 and a pen that costs $1.80. I added the price of the notebooks and the pen together, $2.35 plus $1.80, which is $4.15. He pays with a $20 bill, so I took $4.15 away from $20 and got $15.85, which is his change.",
51: "Sure. Each bag is 2.5 kilograms and there are 4 bags, so I multiplied 2.5 by 4 and got 10 kilograms of rice. The meals are measured in grams, so I changed the 10 kilograms into grams, which is 1000 grams. Then each meal uses 150 grams, so I divided 1000 by 150 and got 6.67. You can only make full meals, so it can make 6 full meals.",
54: "Sure. Mila has 48 stamps and she has 12 more than Noah, so to get Noah's stamps I added 12 to 48 and got 60 stamps. Then Noah has twice as many stamps as Owen, so Owen has half of what Noah has. Half of 60 is 30, so Owen has 30 stamps.",
56: "Sure. The town started with 8,000 people. It grew by 5% in the first year and 5% in the second year, so in two years it grew by 10% altogether. 10% of 8,000 is 800 people, so I added 800 to 8,000 and got 8,800 people after two years.",
58: "Sure. I found her speed first by dividing 3 kilometers by 18 minutes, which is 0.17 kilometers per minute. Then to find the time for 10 kilometers I divided the 10 kilometers by her speed, 0.17, and got 58.8 minutes. That is about 59 minutes when I round it, so it takes her about 59 minutes.",
64: "Sure. On weekdays the cafe opens at 8:30 and closes at 5:00, which is 8 and a half hours, so 8.5 hours. I multiplied 8.5 by the 6 weekdays and got 51 hours. On Saturday it is open from 9 to 2, which is 5 hours, so I added that and got 56 hours a week. It is closed on Sundays so I didn't add anything for Sunday.",
66: "Sure. Each pack has 3 pairs and Jamal needs 11 pairs, so I divided 11 by 3 to see how many packs he needs, and got 3.67. He can't buy part of a pack, so I rounded it up to 4 packs so he has enough socks. 4 packs cost 4 times $10, which is $40.",
70: "Sure. First I found the rookie cards. Jake has 120 cards and 1/4 of them are rookie cards, so 120 divided by 4 is 30 rookie cards. Then 2/3 of them are from this season, which means the other 1/3 are not from this season. 1/3 of 120 is 40 cards, so 40 are not from this season.",
72: "Sure. The deal is buy 2 get 1 free, so for every 2 shirts one shirt is free. Erin has 10 shirts, so I divided 10 by 2 and got 5 free shirts. That means she pays for the other 5 shirts, and each shirt is $9, so 5 times $9 is $45.",
74: "Sure. There are 30 students in the class. 18 students play soccer and 12 play basketball, so I added them and got 30 students who play a sport. The class only has 30 students, so if 30 play a sport then nobody is left, so 0 students play neither sport.",
79: "Sure. The bag of apples weighs 1.2 kilograms and the bag of oranges weighs 1 kilogram. I subtracted to see how much heavier the apples are, 1.2 minus 1, which is 0.2 kilograms. Then I changed 0.2 kilograms into grams, which is 200 grams. So the apple is 200 grams heavier than the orange.",
81: "Sure. Mina saves $5 in the first week and then she saves $3 more every week. So by week 6 she has added $3 six times, which is 6 times 3, 18 dollars. I added the 18 to the $5 from the first week and got $23. So after 6 weeks she has saved $23.",
83: "Sure. Some tables seat 8 people and some seat 12 people, so I found the average of 8 and 12, which is 10 people per table. There are 15 round tables and 10 long tables, so 25 tables. 25 tables times 10 people is 250 seats. There are 265 guests, so I took away 250 and they need 15 more seats.",
87: "Sure. First I found one share of the bill, $132 divided by 6 friends, which is $22. It is one friend's birthday, so the other five friends have to pay her share too. Her share is $22, so I added another $22 onto each of the five friends, 22 plus 22, which makes $44 for each of them.",
88: "Sure. Ms. Lee drives 18 miles to work each day and she works 5 days a week, so I multiplied 18 by 5 and got 90 miles a week. Her car goes 32 miles on one gallon, so I divided 90 by 32 and got 2.8125 gallons. Then gas is $3.60 a gallon, so I multiplied 2.8125 by 3.60 and got $10.13.",
91: "Sure. Job A pays $15 an hour and you work 30 hours, so I multiplied 15 by 30 and got $450. Job B pays $2,100. To see which job pays more I subtracted $450 from $2,100 and got $1,650. So Job B pays $1,650 more.",
93: "Sure. Sam had $120. He wrote a check for $45 and a check for $95, so I took both of them away from $120 and got -$20. Then he deposited $60, so I added 60 to -20 and got $40. At the end he has more than $0, so he doesn't have to pay the $25 fee, and his balance is $40.",
94: "Sure. The bus costs $480 and the tickets cost $12, so I added them together to get the total cost of the trip, which is $492. Then the 30 students share the cost equally, so I divided $492 by 30 and got $16.40 for each student.",
97: "Sure. The mother is 36 now. I divided 36 by 3 and got 12, so when the son is 12 the mother is 3 times older than him. The son is 8 now, so I worked out how long until he is 12, which is 12 minus 8, 4 years. So in 4 years the mother will be 3 times as old as her son.",
98: "Sure. Each tricycle has 3 wheels, so to find the number of tricycles I divided the 47 wheels by 3. I got 15.67, but you can't have part of a tricycle, so I rounded it down to 15. So there are 15 tricycles in the shop.",
}

PROBES = [
    "Can you walk me through your solution?",
    "Could you please walk me through your solution?",
    "Can you talk me through your solution?",
    "Can you tell me how you got your answer?",
    "Could you talk me through your working please?",
    "Can you explain how you got to your answer?",
    "Please walk me through how you solved it.",
    "Could you go through your steps for me?",
]
