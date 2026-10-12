# Failed prompts, grouped by router | failure | oracle model

Biggest groups first; each prompt shows what the router paid for against what the oracle needed.

## nirt | overspend | oracle=mistral-7b-instruct (n=903)

### 853  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.75)  |  oracle: mistral-7b-instruct ($0.0001, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 2350  [mmlu-elementary-mathematics]
router: gpt-4-1106-preview ($0.0008, score 0.00)  |  oracle: mistral-7b-instruct ($0.0000, score 0.00)

> Please answer with the letter of the correct answer.
> 
> What value of y makes y + 2.9 = 11 true?
> A) 8.9
> B) 13.9
> C) 9.1
> D) 8.1
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 2088  [hellaswag]
router: yi-34b-chat ($0.0002, score 1.00)  |  oracle: mistral-7b-instruct ($0.0001, score 1.00)

> [header] How to diagnose and treat salpingitis [title] Understand that women with salpingitis sometimes have no symptoms. [step] In some cases, salpingitis causes no noticeable symptoms. Therefore, you may not discover that you have the condition until your reproductive organs are examined for another reason, such as infertility.
> A) [title] Pay attention to abnormal vaginal discharge. [step] Some vaginal discharge is normal and healthy, but if you notice any changes in the amount, color, consistency, or smell, you should make an appointment with your gynecologist.
> B) [title] Recognize when you...

## nirt | overspend | oracle=wizardlm-13b-v1.2 (n=688)

### 2802  [mmlu-international-law]
router: yi-34b-chat ($0.0001, score 1.00)  |  oracle: wizardlm-13b-v1.2 ($0.0000, score 1.00)

> Please answer with the letter of the correct answer.
> 
> What kind of rights do coastal States enjoy over their continental shelf?
> A) The coastal State exercise sovereign rights over its continental shelf for the purpose of exploiting its fishing resources
> B) The coastal State may exercise sovereign rights over its continental shelf only upon its declaration
> C) The coastal State enjoy ipso facto and ab initio sovereign rights over its continental shelf for the purpose of exploring it and exploiting its natural resources
> D) The coastal State may exercise only limited rights over its continental sh...

### 154  [arc-challenge]
router: yi-34b-chat ($0.0001, score 1.00)  |  oracle: wizardlm-13b-v1.2 ($0.0000, score 1.00)

> Every night, Johnny reads next to a lamp that is lit by an incandescent light bulb. Which is the best indication that the transfer of electrical energy to the lamp is inefficient?
> A) The lamp uses copper wires.
> B) The bulb gets hot.
> C) The lamp can be turned off at a switch.
> D) The bulb produces light.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

### 36  [arc-challenge]
router: yi-34b-chat ($0.0001, score 1.00)  |  oracle: wizardlm-13b-v1.2 ($0.0000, score 1.00)

> Many endangered animals are captured, tagged, and then released back into their environment. The tag may be a piece of coded plastic or metal that is placed somewhere safe on the body. Which is the most likely reason to tag an animal?
> A) to monitor their eating habits
> B) to monitor their reproductive habits
> C) to monitor their roaming habits
> D) to monitor their sleeping habits
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

## nirt | overspend | oracle=mixtral-8x7b-instruct (n=218)

### 3273  [mmlu-professional-law]
router: gpt-4-1106-preview ($0.0031, score 1.00)  |  oracle: mixtral-8x7b-instruct ($0.0002, score 1.00)

> Please answer with the letter of the correct answer.
> 
> An appliance store was using part of a public alley to unload its trucks. There were no warning signs keeping pedestrians out or warning of unloading trucks. A pedestrian walked through the general vicinity but he avoided the unloading area. One of the workers pulled the wrong lever and a load of commercial refrigerators crashed into the street, causing a rumbling tremor in the road. The tremor caused a heavy box to fall off of a parked truck that was about 30 yards away from the loading dock. It fell on the pedestrian, causing serious inju...

### 1467  [hellaswag]
router: claude-instant-v1 ($0.0002, score 1.00)  |  oracle: mixtral-8x7b-instruct ($0.0002, score 1.00)

> [header] How to work out salary increase percentage [title] Subtract your old salary from your new salary. [step] Say you made $45 , 000 per year at your old job and that you accepted a new position making $50 , 000 per year. This means you would take $50 , 000 and subtract $45 , 000.
> A) The remaining $70 , 000 would equal $100 , 000 in your new salary. [substeps] If you have less in your previous salary than you did in your current position, you should subtract $100 , 000 from your current salary and make the difference.
> B) You would subtract $50 , 000 from the $45 , 000 to get your new salar...

### 952  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.75)  |  oracle: mixtral-8x7b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## nirt | quality_loss | oracle=mistral-7b-instruct (n=206)

### 1145  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: mistral-7b-instruct ($0.0000, score 1.00)

> Two people from the group are dressed in winter gear, go down the slope in their tubes. the other two group members
> A) follow them in their tubes as they go down the mountain slope.
> B) go down the hill together, one appears to be the right ddler and the other is the espeaker.
> C) are dressed in winter gear, the first group grabs ropes and walks together to the water and falls.
> D) get into the water in their tubes and start paddling down the hill.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

### 3429  [mmlu-professional-psychology]
router: gpt-4-1106-preview ($0.0016, score 0.00)  |  oracle: mistral-7b-instruct ($0.0000, score 1.00)

> Please answer with the letter of the correct answer.
> 
> An evaluation of a particular treatment fails to show an overall effect. However, the experimenter does find thar the most severely disturbed do show considerable improvement, and therefore concludes that the treatment is effective with extreme cases. This conclusion
> A) fails to take into account regression toward the mean
> B) fails to take into account the fact thar the “placebo effect” is most pronounced in severely disturbed patients
> C) is valid if t test comparing severely disturbed patients with less severely disturbed patients is signi...

### 51  [arc-challenge]
router: yi-34b-chat ($0.0000, score 0.00)  |  oracle: mistral-7b-instruct ($0.0000, score 1.00)

> How many valence electrons does selenium have?
> A) 6
> B) 3
> C) 8
> D) 5
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

## nirt | quality_loss | oracle=wizardlm-13b-v1.2 (n=175)

### 1100  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.50)  |  oracle: wizardlm-13b-v1.2 ($0.0002, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 3342  [mmlu-professional-law]
router: gpt-4-1106-preview ($0.0033, score 0.00)  |  oracle: wizardlm-13b-v1.2 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> An insured business owner stored valuable property in an insured warehouse owned by the business. During night time hours, when there were no employees in the warehouse, the building and its contents burned to the ground by a raging fire. The fire department ruled that the fire had an incendiary origin. The fire report, however, did not identify a perpetrator. The insurer responded by rejecting coverage and accusing the business owner of starting the fire. The business owner sued the insurance company in federal court for the policy proceed...

### 1574  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: wizardlm-13b-v1.2 ($0.0000, score 1.00)

> A man sits a piano with his hands placed on the keys. The man begins playing the piano. credits
> A) roll and he wipes his hands.
> B) will be shown on the screen.
> C) roll in the screen.
> D) begin rolling as the man continues to play the piano.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

## nirt | quality_loss | oracle=mixtral-8x7b-instruct (n=171)

### 823  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.25)  |  oracle: mixtral-8x7b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 3117  [mmlu-philosophy]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: mixtral-8x7b-instruct ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> What was Clarence Darrow's belief about life's meaning?
> A) Life was uncertain but ultimately rewarding.
> B) Life was hard but worthwhile.
> C) Life was meaningful for most people.
> D) Life was not worthwhile.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 921  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.50)  |  oracle: mixtral-8x7b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## nirt | quality_loss | oracle=claude-instant-v1 (n=87)

### 292  [consensus_summary]
router: yi-34b-chat ($0.0003, score 0.75)  |  oracle: claude-instant-v1 ($0.0002, score 1.00)

> Generate a brief answer using only the provided claims, with no personal opinions or outside knowledge. If there is no answer based on the claims, write 'N-A'.
> 
> claim: Dietary fish oil produces changes in cardiac function that might contribute to cardiovascular health benefits in humans and does so by modifying cardiac membranes within a dose range achievable in the human diet.
> claim: The results suggest that dietary supplementation with fish oil may be beneficial in reducing myocardial damage associated with coronary artery thrombosis.
> claim: In conclusion, fish oil supplementation has a bene...

### 1194  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: claude-instant-v1 ($0.0002, score 1.00)

> [header] How to clean a driveway [title] Clear out your driveway. [step] Remove any loose materials, cars, dirt, or stone from your driveway. If your driveway touches any doors or walls, cover them with cardboard, a tarp, or some painter's tape.
> A) Put on gloves to avoid scratching your hands. [title] Mark the spot on your driveway using a pencil and a utility knife.
> B) This will protect them from any debris that may fly around while you are pressure washing. [substeps] Check to see which end of your driveway is the highest and what direction the water will flow while you are washing.
> C) [subs...

### 2085  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: claude-instant-v1 ($0.0002, score 1.00)

> [header] How to find a parking spot [title] Download the mobile parking app that is best for you. [step] Open the app store on your smartphone and then open the navigation category. Look for relevant parking apps that meet your key criteria.
> A) [title] Look over the categories for available parking spots. [step] Once you've opened the app, navigate to the categories for your near and far-off parking spot.
> B) For instance, you may want to choose an app that allows you to pre-pay for parking or sends notifications when your parking is about to expire. If discounts are a priority, some apps have ...

## nirt | quality_loss | oracle=gpt-3.5-turbo-1106 (n=82)

### 1180  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: gpt-3.5-turbo-1106 ($0.0002, score 1.00)

> [header] How to obtain the percent complete numerical value from an audible audiobook from the audible for ios app [title] Turn on the device and get to your home screen. [title] Make sure that you've installed and/or updated audible's official ios app. [step] Check the updates tab on the appstore integrated app to ensure you are completely updated.
> A) [title] Go to https: // www.audio.google.com in a web browser. [title] Click on the menu at the top of the screen and select " appaller.
> B) [title] Browse for and then install audible. [step] Below the initial listings, you should see a prompt t...

### 186  [bias_detection]
router: claude-instant-v1 ($0.0003, score 0.00)  |  oracle: gpt-3.5-turbo-1106 ($0.0003, score 1.00)

> Classify the following sentence of a news article as fact (statment free of rephrasing, interpretation, opinions, and emotions), opinion (expression of a personal view, judgement, appraisal, opinion, or interpretation), claim (assertion of unverified information, rephrased facts, or affirmation of opinions), argument (data, information, reference, opinion, or narrative used to support a claim), data (raw data or statistics, must incluide the source which cant be a person, and must exclude any interpretation of the data), quote (direct quote from a person or a document), narrative (a story, acc...

### 2078  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: gpt-3.5-turbo-1106 ($0.0001, score 1.00)

> He then begins to rub the frame with a blue rag. As he does this he explains his methods. using the same rag he
> A) rubs over the railings of a wall.
> B) then shuffles the wax off of the first bar.
> C) puts white foam around the edges of the frame to make a square frame.
> D) moves to the other pane to clean the frame.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

## nirt | overspend | oracle=yi-34b-chat (n=62)

### 2378  [mmlu-elementary-mathematics]
router: gpt-4-1106-preview ($0.0010, score 1.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> A tournament has 393 players with 3 players on each team. Which expression shows how to find the number of teams in the tournament?
> A) 393 + 3
> B) 393 – 3
> C) 393 × 3
> D) 393 ÷ 3
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 2449  [mmlu-high-school-chemistry]
router: gpt-4-1106-preview ($0.0011, score 1.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> PCl3(g) + Cl2(g) ↔ PCl5(g) ΔH = -92.5 kJ/mol In which of the following ways could the reaction above be manipulated to create more product?
> A) Decreasing the concentration of PCl3
> B) None of the above
> C) Increasing the pressure
> D) Increasing the temperature
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 2565  [mmlu-high-school-macroeconomics]
router: gpt-4-1106-preview ($0.0010, score 1.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Stagflation is the result of
> A) a rightward shift in the aggregate supply curve.
> B) a leftward shift in both the aggregate supply and aggregate demand curves.
> C) a leftward shift in the aggregate demand curve.
> D) a leftward shift in the aggregate supply curve.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

## nirt | quality_loss | oracle=gpt-4-1106-preview (n=51)

### 1619  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: gpt-4-1106-preview ($0.0028, score 1.00)

> [header] How to remove an item from a list in foursquare [title] Open up foursquare. [title] Search for the item you'd like to remove from the list. [step] You can tap the " lists " tab at the bottom of the screen, tap the list, then tap the item from the list.
> A) [title] Tap " shop " next to the item you'd like to remove. [title] Find and tap " get rid of " next to its name.
> B) [title] Swipe up and down on the item. [step] The item will slide right, making it much easier to remove.
> C) [title] Tap the info tab under the business' name and category/city. [title] Scroll up until you see the butt...

### 423  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.25)  |  oracle: gpt-4-1106-preview ($0.0116, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 347  [grade-school-math]
router: claude-instant-v1 ($0.0004, score 0.25)  |  oracle: gpt-4-1106-preview ($0.0106, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## nirt | quality_loss | oracle=yi-34b-chat (n=50)

### 2332  [mmlu-econometrics]
router: gpt-4-1106-preview ($0.0013, score 0.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Which one of the following is a disadvantage of the general to specific or "LSE" ("Hendry") approach to building econometric models, relative to the specific to general approach?
> A) If the initial model is mis-specified, all subsequent steps will be invalid.
> B) Some variables may be excluded at the first stage leading to coefficient biases
> C) The final model may lack theoretical interpretation
> D) The final model may be statistically inadequate
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 708  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.25)  |  oracle: yi-34b-chat ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 691  [grade-school-math]
router: claude-instant-v1 ($0.0003, score 0.25)  |  oracle: yi-34b-chat ($0.0005, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## nirt | quality_loss | oracle=claude-v2 (n=37)

### 1982  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: claude-v2 ($0.0008, score 1.00)

> A group of kids are on a track by the field. They pose for pictures before jumping onto a table. they
> A) then start a game of ping pong.
> B) play racquetball for a bit before jumping back to play again.
> C) sit and play dodgeball as the coach walks by.
> D) then sit on the track for pictures again.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

### 1813  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: claude-v2 ($0.0020, score 1.00)

> [header] How to repair a scrape on faux leather shoes [title] Wipe down the area with a soft cloth and warm water. [step] Then dab the area with a little distilled white vinegar. Treat a small part of the affected area with the vinegar.
> A) [title] Add a bit of water, salt, baking soda and a bit of dish soap to the mixture and allow to sit for 1 to 3 minutes, then apply a bit more dish soap. [step] This will prevent the stain from washing off.
> B) Repeat this several times, then wrap it in a cloth or a towel to keep it warm. [substeps] The vinegar will soften the leather during this step.
> C) Rep...

### 1357  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: claude-v2 ($0.0021, score 1.00)

> [header] How to apply for paternity leave [title] Read your employee handbook. [step] See if your company policies include the option of taking paternity leave. Keep in mind that if your employer offers any kind of maternity leave for new mothers, it is required by law to offer comparable paternity leave for new fathers.
> A) Read your employer handbook closely to see if you and your employer have decided on the frequency and duration of a designated paternity leave. This is important, as you should bring this information with you to any applications.
> B) You will have no legal recourse if you ar...

## nirt | quality_loss | oracle=llama-2-70b-chat (n=37)

### 1705  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: llama-2-70b-chat ($0.0001, score 1.00)

> A man is seen speaking to the camera and standing in front of a table with various tools laid out in front of him. He then is seen wearing a hat, holding a mug, and using a knife around it. he
> A) continues the process until it eventually cuts off and he finally cups a cup and pours it into the mug.
> B) wipes mud all over the table and moves there tools all around.
> C) then uses the knife along a window and is shown several times sharpening the knife and speaking to the camera.
> D) turns the knife around and demonstrates how it works, cutting the lemons out with a small cutter and shashing them in...

### 1622  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: llama-2-70b-chat ($0.0003, score 1.00)

> [header] How to power wash a house [title] Cover all exterior electrical outlets. [step] Apply a piece of duct tape to cover and seal the outlets, even if the outlets are already equipped with spring tension covers. The tape will help prevent any water and detergent from leaking into the outlets.
> A) [substeps] If the interior of the house is covered with duct tape, pad the area around the outlets with masking tape to seal them even if they are not marked. [title] Check that the wall and door hardware are completely covered with duct tape.
> B) [substeps] If the home is in an area that's prone to...

### 1472  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: llama-2-70b-chat ($0.0001, score 1.00)

> A person is shown in various shots close up as well as leaves and him stretching. the man
> A) shakes his hair to show how it looks and then begins to rub his hands all over it and finally putting some lotion on.
> B) walks away while speaking to the camera and a person holds up a photo.
> C) moves several bags and mulch around the tree.
> D) is then seen running through a park while looking off into the distance.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

## nirt | quality_loss | oracle=claude-v1 (n=30)

### 275  [consensus_summary]
router: claude-instant-v1 ($0.0002, score 0.00)  |  oracle: claude-v1 ($0.0016, score 1.00)

> Generate a brief answer using only the provided claims, with no personal opinions or outside knowledge. If there is no answer based on the claims, write 'N-A'.
> 
> claim: Results confirm that money is the main cause, but we found that governmental budget deficit through an increase in money supply also indirectly affect inflation.
> claim: Modest inflation boosts growth rate and welfare
> claim: Allowing for a time-varying treatment effect, we show that Inflation Targeting (IT) effectively lowers inflation for both developed and developing countries.
> claim: It is concluded in this paper that, althoug...

### 3421  [mmlu-professional-psychology]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: claude-v1 ($0.0007, score 1.00)

> Please answer with the letter of the correct answer.
> 
> On the WAIS-III, a Verbal IQ that is substantially higher than the Performance IQ is most suggestive of which of the following?
> A) low socioeconomic status
> B) depression
> C) delinquency
> D) a learning disability
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 607  [grade-school-math]
router: claude-instant-v1 ($0.0009, score 0.25)  |  oracle: claude-v1 ($0.0056, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## nirt | overspend | oracle=llama-2-70b-chat (n=29)

### 498  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.75)  |  oracle: llama-2-70b-chat ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 1003  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.75)  |  oracle: llama-2-70b-chat ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 1046  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.75)  |  oracle: llama-2-70b-chat ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## nirt | overspend | oracle=code-llama-34b-instruct (n=25)

### 462  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.75)  |  oracle: code-llama-34b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 387  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.75)  |  oracle: code-llama-34b-instruct ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 995  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.75)  |  oracle: code-llama-34b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## nirt | overspend | oracle=claude-instant-v1 (n=20)

### 3509  [mmlu-virology]
router: gpt-4-1106-preview ($0.0008, score 1.00)  |  oracle: claude-instant-v1 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> The characteristic of a generalized epidemic is:
> A) High rate in STD clinic attendees
> B) High rates in drug users
> C) High rates in MSM
> D) High rates in pregnant women
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 1977  [hellaswag]
router: gpt-4-1106-preview ($0.0009, score 1.00)  |  oracle: claude-instant-v1 ($0.0001, score 1.00)

> A man is interviewed while people watch or walk past in the background. Two men are interviewed in the same setting. a woman
> A) is interviewed and two men are interviewed.
> B) is interviewed while other people are interviewed.
> C) talks to the camera.
> D) is interviewed in the same setting.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

### 3515  [mmlu-virology]
router: gpt-4-1106-preview ($0.0009, score 1.00)  |  oracle: claude-instant-v1 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> An updated version of classification shows the parvovirus family as having how many genera?
> A) Having 5 genera
> B) Only a single virus
> C) A very large number of virus types
> D) Only three viruses
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

## nirt | overspend | oracle=gpt-3.5-turbo-1106 (n=18)

### 244  [chinese_zodiac]
router: gpt-4-1106-preview ($0.0011, score 1.00)  |  oracle: gpt-3.5-turbo-1106 ($0.0001, score 1.00)

> You are an assistant who guesses the exact age using the chinese animal zodiac signs and age zones of the zodiac. A user wishing to obtain the subject's exact age provides the zodiac animal zodiac and the estimated age range of the subject, resulting in the year of birth. Answer only the 4-digit year of birth and do not provide any other explanation. If the current year, zodiac sign, and approximate age range are provided, please translate and print only the input without additional sentences.
> 
> 2024, ox, 50s

### 773  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.75)  |  oracle: gpt-3.5-turbo-1106 ($0.0005, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 968  [grade-school-math]
router: claude-instant-v1 ($0.0010, score 0.75)  |  oracle: gpt-3.5-turbo-1106 ($0.0006, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## nirt | quality_loss | oracle=code-llama-34b-instruct (n=17)

### 279  [consensus_summary]
router: claude-instant-v1 ($0.0002, score 0.25)  |  oracle: code-llama-34b-instruct ($0.0002, score 1.00)

> Generate a brief answer using only the provided claims, with no personal opinions or outside knowledge. If there is no answer based on the claims, write 'N-A'.
> 
> claim: Anger, tension, and fatigue increased according with training load, suggesting that the training exercise led participants into a negative psychological state.
> claim: Over-training was accompanied by alterations in the lipid profile, which appeared to be the consequence of over-training.
> claim: The negative effects of intensive training at a young age were outweighed by the many social, psychological and health benefits that a s...

### 640  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.25)  |  oracle: code-llama-34b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 715  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.50)  |  oracle: code-llama-34b-instruct ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | overspend | oracle=mistral-7b-instruct (n=915)

### 888  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.75)  |  oracle: mistral-7b-instruct ($0.0001, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 2843  [mmlu-machine-learning]
router: gpt-4-1106-preview ($0.0008, score 1.00)  |  oracle: mistral-7b-instruct ($0.0000, score 1.00)

> Please answer with the letter of the correct answer.
> 
> As the number of training examples goes to infinity, your model trained on that data will have:
> A) Lower variance
> B) Higher variance
> C) None of the above
> D) Same variance
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 449  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.75)  |  oracle: mistral-7b-instruct ($0.0001, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | overspend | oracle=wizardlm-13b-v1.2 (n=689)

### 81  [arc-challenge]
router: yi-34b-chat ($0.0001, score 1.00)  |  oracle: wizardlm-13b-v1.2 ($0.0000, score 1.00)

> Heat was applied to a substance. First the substance expanded, and then it melted. Which state of matter did the substance start off as?
> A) Plasma
> B) Solid
> C) Liquid
> D) Gas
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

### 3272  [mmlu-professional-law]
router: yi-34b-chat ($0.0003, score 1.00)  |  oracle: wizardlm-13b-v1.2 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> A seller listed real property for $500,000, which was set to expire on Aug.1. The current lessee of the premises, who had made substantial improvements to the property, wanted to buy at the full list price but needed extra time. He paid $10,000 for an option to be added to the lease, giving him an option to buy on or before Sept. 1. The option stated that all notices required under it "shall be deemed given if given in writing and delivered personally, delivered by commercial delivery service, or by certified mail return receipt requested, ...

### 717  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.75)  |  oracle: wizardlm-13b-v1.2 ($0.0002, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | overspend | oracle=mixtral-8x7b-instruct (n=205)

### 792  [grade-school-math]
router: claude-instant-v1 ($0.0008, score 0.75)  |  oracle: mixtral-8x7b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 3302  [mmlu-professional-law]
router: yi-34b-chat ($0.0002, score 1.00)  |  oracle: mixtral-8x7b-instruct ($0.0002, score 1.00)

> Please answer with the letter of the correct answer.
> 
> In an effort to improve the health care of cancer patients, a state's legislature has enacted a law providing public funds to assist privately owned hospitals. According to the law, every hospital in the state would receive $5,000 annually for each cancer patient who was admitted to that hospital, and whose period of hospitalization exceeded one week. Although this bill was initially opposed by several churches and other organizations, the state's legislature re-drafted the bill in a compromise effort to appease the opposition. In its final...

### 234  [chinese_zodiac]
router: gpt-4-1106-preview ($0.0011, score 1.00)  |  oracle: mixtral-8x7b-instruct ($0.0001, score 1.00)

> You are an assistant who guesses the exact age using the chinese animal zodiac signs and age zones of the zodiac. A user wishing to obtain the subject's exact age provides the zodiac animal zodiac and the estimated age range of the subject, resulting in the year of birth. Answer only the 4-digit year of birth and do not provide any other explanation. If the current year, zodiac sign, and approximate age range are provided, please translate and print only the input without additional sentences.
> 
> 2023, goat, 30s

## zoib | quality_loss | oracle=mistral-7b-instruct (n=194)

### 2365  [mmlu-elementary-mathematics]
router: gpt-4-1106-preview ($0.0009, score 0.00)  |  oracle: mistral-7b-instruct ($0.0000, score 1.00)

> Please answer with the letter of the correct answer.
> 
> The value of the digit 5 in 24,513 is how many times the value of the digit 5 in 357?
> A) 10
> B) 10,000
> C) 1,000
> D) 100
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 3518  [mmlu-virology]
router: mixtral-8x7b-instruct ($0.0000, score 0.00)  |  oracle: mistral-7b-instruct ($0.0000, score 1.00)

> Please answer with the letter of the correct answer.
> 
> The major issue for HIV infection among the Latino population is:
> A) All of the above
> B) Late detection of HIV status
> C) Cultural beliefs
> D) Immigration
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 1199  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: mistral-7b-instruct ($0.0000, score 1.00)

> He is very into this song that he is playing, he sits there very concentrated just playing his song. he
> A) is an older man with many years of experience on the harmonica so for his it is a passion that he takes to heart.
> B) stops and looks down at his hands.
> C) doesn't show any other gook as he continues to show how it should be.
> D) looks to the right and speaks to the camera.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

## zoib | quality_loss | oracle=wizardlm-13b-v1.2 (n=174)

### 1260  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: wizardlm-13b-v1.2 ($0.0000, score 1.00)

> A guy comes onto the screen to introduce a video about his character on a kids show. he
> A) shows off a series of leads on how he becomes a professional wrestler.
> B) begins by introducing a girl named jennifer, who is walking with her dad behind her.
> C) goes to an indoor pool, and shows how to do a splash dunk.
> D) begins to show several different clips from the show.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

### 3322  [mmlu-professional-law]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: wizardlm-13b-v1.2 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> A nightclub owner applied for a required zoning permit to open a nude-dancing nightclub in the theater district of a city. An organization of influential city residents began an intensive lobbying effort to persuade the city council to deny the owner a permit to operate any type of nude- dancing facility at any time or in any place in the city. The owner has sued the city in an appropriate federal court, seeking an injunction that would prohibit the city council from considering the organization's views, on the ground that if the organizati...

### 1748  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: wizardlm-13b-v1.2 ($0.0001, score 1.00)

> [header] How to settle an argument with your wife [title] Stay in the present. [step] Arguments are often the result of underlying tensions in a relationship. If you want to settle an argument effectively you must be willing to communicate with your wife.
> A) When your wife is upset, she often grabs her husband by the collar, trying to get his attention by hitting and spanking him until she calms down. This sort of behavior leads to hurt feelings, stress and anger and even physical arguments.
> B) This means staying in the present moment and avoiding discussing past conflicts. [substeps] In an ar...

## zoib | quality_loss | oracle=mixtral-8x7b-instruct (n=156)

### 524  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.50)  |  oracle: mixtral-8x7b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 843  [grade-school-math]
router: claude-instant-v1 ($0.0010, score 0.75)  |  oracle: mixtral-8x7b-instruct ($0.0003, score 1.00)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 631  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.50)  |  oracle: mixtral-8x7b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | quality_loss | oracle=claude-instant-v1 (n=109)

### 2923  [mmlu-miscellaneous]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: claude-instant-v1 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> How many $100 bills does it take to equal one million dollars?
> A) five thousand
> B) one hundred thousand
> C) one thousand
> D) ten thousand
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 303  [consensus_summary]
router: mixtral-8x7b-instruct ($0.0002, score 0.00)  |  oracle: claude-instant-v1 ($0.0002, score 1.00)

> Generate a brief answer using only the provided claims, with no personal opinions or outside knowledge. If there is no answer based on the claims, write 'N-A'.
> 
> claim: Here, the authors show that when retinal mechanisms of color are impaired, memory has a paradoxical impact on color appearance that is selective for faces, providing evidence that color contributes to face encoding and social communication.
> claim: The results indicate that color coding increased retention and transfer performance.
> claim: We may speculate that color ensures preferential access to or retrieval from memory when dis...

### 2511  [mmlu-high-school-geography]
router: mixtral-8x7b-instruct ($0.0000, score 0.00)  |  oracle: claude-instant-v1 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> All of the following are non-basic jobs in the city EXCEPT
> A) dry cleaner.
> B) social worker.
> C) garment factory worker.
> D) city firefighter.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

## zoib | quality_loss | oracle=gpt-3.5-turbo-1106 (n=82)

### 1203  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: gpt-3.5-turbo-1106 ($0.0003, score 1.00)

> [header] How to iron a tie [title] Set up the ironing board. [step] Put the board on a flat surface so that it is steady and will not flip over while you are ironing. Make sure that it is clean and dry before setting your tie down.
> A) [title] Place your iron at a 90-degree angle at least 12 inches (30 cm) away from the four corners of the fabric. [step] Focus the iron on the right corner of the fabric to cover the stitch; the edge where the stitch meets the strip of material running horizontally along the top of the fabric.
> B) Always set the ironing board up so that there is an opening in it, ...

### 202  [Chinese_character_riddles]
router: mixtral-8x7b-instruct ($0.0002, score 0.00)  |  oracle: gpt-3.5-turbo-1106 ($0.0002, score 1.00)

> 猜字谜，根据我给的描述猜出一个字(请从汉字的字形、发音、意义以及字的拆分组合等角度考虑)。首先提供你的推理，然后提供用英文方括号括[]起来的最终答案。
> 
> 地皮也刮去

### 2078  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: gpt-3.5-turbo-1106 ($0.0001, score 1.00)

> He then begins to rub the frame with a blue rag. As he does this he explains his methods. using the same rag he
> A) rubs over the railings of a wall.
> B) then shuffles the wax off of the first bar.
> C) puts white foam around the edges of the frame to make a square frame.
> D) moves to the other pane to clean the frame.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

## zoib | quality_loss | oracle=yi-34b-chat (n=62)

### 2277  [mmlu-college-medicine]
router: mixtral-8x7b-instruct ($0.0000, score 0.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Glycogen breakdown in exercising muscle is activated by:
> A) increased pH.
> B) cortisol.
> C) insulin.
> D) none of the above.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 2332  [mmlu-econometrics]
router: gpt-3.5-turbo-1106 ($0.0001, score 0.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Which one of the following is a disadvantage of the general to specific or "LSE" ("Hendry") approach to building econometric models, relative to the specific to general approach?
> A) If the initial model is mis-specified, all subsequent steps will be invalid.
> B) Some variables may be excluded at the first stage leading to coefficient biases
> C) The final model may lack theoretical interpretation
> D) The final model may be statistically inadequate
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 2348  [mmlu-elementary-mathematics]
router: gpt-4-1106-preview ($0.0010, score 0.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Which of the following numbers is between 2,329,500 and 2,598,100?
> A) 2,249,550
> B) 2,327,900
> C) 2,589,200
> D) 2,329,333
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

## zoib | quality_loss | oracle=gpt-4-1106-preview (n=58)

### 2070  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: gpt-4-1106-preview ($0.0012, score 1.00)

> . a young female dressed in a white jersey and purple shorts
> A) is sitting on the courtyard floor holding a violin and smiling to the camera, talking.
> B) is standing on the edge of a pool stretching.
> C) is putting on her weightlifting gear and is preparing to begin barbiedoll lifting.
> D) is standing outside wearing a karate uniform trying to do her basic karate moves in a large open gym that has many wooden walls.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

### 2058  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: gpt-4-1106-preview ($0.0022, score 1.00)

> [header] How to change visualizations in winamp [title] Launch winamp. [step] Do this by double clicking on the winamp icon (lightning bolt) on your desktop or in your applications folder. [title] Click on " view " on the menu toolbar.
> A) [step] Then select " view background " from the menu. [title] Click on the " custom background " button.
> B) [title] Click on " left-click ". [step] This is located in the sidebar at the bottom of your mac screen.
> C) [step] Upon opening this dialog box, you will see a list of all your saved visualizations. Scroll down to one of the images you want to view and ...

### 2522  [mmlu-high-school-government-and-politics]
router: mixtral-8x7b-instruct ($0.0001, score 0.00)  |  oracle: gpt-4-1106-preview ($0.0010, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Which of the following CANNOT limit the influence of the federal courts?
> A) A constitutional amendment to the contrary of a court decision
> B) A national election recalling an unpopular judge
> C) The impeachment of a federal judge
> D) A governor's failure to enforce a court decision
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

## zoib | overspend | oracle=yi-34b-chat (n=52)

### 2343  [mmlu-electrical-engineering]
router: gpt-4-1106-preview ($0.0008, score 1.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Speed control by variation of field flux results in
> A) none of the above.
> B) constant torque drive.
> C) constant power drive.
> D) variable power drive.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 2715  [mmlu-high-school-statistics]
router: gpt-4-1106-preview ($0.0013, score 1.00)  |  oracle: yi-34b-chat ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> The number of days it takes to build a new house has a variance of 386. A sample of 40 new homes shows an average building time of 83 days. With what confidence can we assert that the average building time for a new house is between 80 and 90 days?
> A) 15.4%
> B) 20.0%
> C) 17.8%
> D) 82.1%
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 901  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.75)  |  oracle: yi-34b-chat ($0.0002, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | quality_loss | oracle=llama-2-70b-chat (n=38)

### 1677  [hellaswag]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: llama-2-70b-chat ($0.0001, score 1.00)

> The left team makes a goal and get excited. They make another shot and the man in white begins to freak out. the man in black
> A) kneels down and sends a shot to the opposite team and they try to join him in his run to get away from their goal.
> B) runs across the court with a rubber pole.
> C) is excited and everyone starts hugging everyone else.
> D) goes next and grabs his head with both hands.
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation.
> Answer:

### 3565  [winogrande]
router: yi-34b-chat ($0.0000, score 0.00)  |  oracle: llama-2-70b-chat ($0.0000, score 1.00)

> The drink was rich in caffeine and low in sugar because the _ was not enough to keep the drinker awake.
> A) caffeine
> B) sugar
> Print only a single choice  from "A" or "B" without explanation. Answer:

### 1389  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: llama-2-70b-chat ($0.0002, score 1.00)

> [header] How to apply basic stage makeup (for women ) [title] Start with a layer of foundation, applied with a latex sponge. [step] You want to achieve a nice even coverage. For lighter skin tones, choose a foundation a few shades darker than your natural tone as this will help combat the bleaching effect of stage lights.
> A) For hair, experiment with bringing up to 2 shades lighter, then using concealer, hair dye and mascara. [substeps] The foundation should be thin, concealable and waterproof.
> B) [title] Before setting the foundation, make sure that any creases around the eyes have been worke...

## zoib | quality_loss | oracle=claude-v2 (n=35)

### 2407  [mmlu-global-facts]
router: mixtral-8x7b-instruct ($0.0000, score 0.00)  |  oracle: claude-v2 ($0.0006, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Which of the following is not an element of the redistribution-with-growth policy approach?
> A) progressive taxation
> B) land reform
> C) increased access to education
> D) minimum wage legislation
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 1813  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: claude-v2 ($0.0020, score 1.00)

> [header] How to repair a scrape on faux leather shoes [title] Wipe down the area with a soft cloth and warm water. [step] Then dab the area with a little distilled white vinegar. Treat a small part of the affected area with the vinegar.
> A) [title] Add a bit of water, salt, baking soda and a bit of dish soap to the mixture and allow to sit for 1 to 3 minutes, then apply a bit more dish soap. [step] This will prevent the stain from washing off.
> B) Repeat this several times, then wrap it in a cloth or a towel to keep it warm. [substeps] The vinegar will soften the leather during this step.
> C) Rep...

### 1357  [hellaswag]
router: yi-34b-chat ($0.0002, score 0.00)  |  oracle: claude-v2 ($0.0021, score 1.00)

> [header] How to apply for paternity leave [title] Read your employee handbook. [step] See if your company policies include the option of taking paternity leave. Keep in mind that if your employer offers any kind of maternity leave for new mothers, it is required by law to offer comparable paternity leave for new fathers.
> A) Read your employer handbook closely to see if you and your employer have decided on the frequency and duration of a designated paternity leave. This is important, as you should bring this information with you to any applications.
> B) You will have no legal recourse if you ar...

## zoib | quality_loss | oracle=claude-v1 (n=33)

### 467  [grade-school-math]
router: claude-instant-v1 ($0.0003, score 0.25)  |  oracle: claude-v1 ($0.0041, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 2363  [mmlu-elementary-mathematics]
router: yi-34b-chat ($0.0001, score 0.00)  |  oracle: claude-v1 ($0.0006, score 1.00)

> Please answer with the letter of the correct answer.
> 
> What is the supplement of a 40° angle?
> A) 50°
> B) 140°
> C) 130°
> D) 220°
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 3093  [mmlu-nutrition]
router: mixtral-8x7b-instruct ($0.0000, score 0.00)  |  oracle: claude-v1 ($0.0006, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Which of the following vitamins provides the coenzyme for reduction reactions in fatty acid synthesis?
> 
> A) Vitamin B6
> B) Niacin
> C) Folate
> D) Riboflavin
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

## zoib | overspend | oracle=llama-2-70b-chat (n=28)

### 498  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.75)  |  oracle: llama-2-70b-chat ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 1046  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.75)  |  oracle: llama-2-70b-chat ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 822  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.75)  |  oracle: llama-2-70b-chat ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | overspend | oracle=code-llama-34b-instruct (n=26)

### 387  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.75)  |  oracle: code-llama-34b-instruct ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 2097  [mbpp]
router: gpt-3.5-turbo-1106 ($0.0004, score 1.00)  |  oracle: code-llama-34b-instruct ($0.0003, score 1.00)

> Write a function to determine if the sum of the divisors of two integers are the same.

### 841  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.75)  |  oracle: code-llama-34b-instruct ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | overspend | oracle=gpt-3.5-turbo-1106 (n=17)

### 239  [chinese_zodiac]
router: gpt-4-1106-preview ($0.0011, score 1.00)  |  oracle: gpt-3.5-turbo-1106 ($0.0001, score 1.00)

> You are an assistant who guesses the exact age using the chinese animal zodiac signs and age zones of the zodiac. A user wishing to obtain the subject's exact age provides the zodiac animal zodiac and the estimated age range of the subject, resulting in the year of birth. Answer only the 4-digit year of birth and do not provide any other explanation. If the current year, zodiac sign, and approximate age range are provided, please translate and print only the input without additional sentences.
> 
> 2022, rabbit, 80s

### 678  [grade-school-math]
router: claude-instant-v1 ($0.0007, score 0.75)  |  oracle: gpt-3.5-turbo-1106 ($0.0006, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 780  [grade-school-math]
router: claude-instant-v1 ($0.0009, score 0.75)  |  oracle: gpt-3.5-turbo-1106 ($0.0006, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | quality_loss | oracle=code-llama-34b-instruct (n=16)

### 279  [consensus_summary]
router: llama-2-70b-chat ($0.0002, score 0.75)  |  oracle: code-llama-34b-instruct ($0.0002, score 1.00)

> Generate a brief answer using only the provided claims, with no personal opinions or outside knowledge. If there is no answer based on the claims, write 'N-A'.
> 
> claim: Anger, tension, and fatigue increased according with training load, suggesting that the training exercise led participants into a negative psychological state.
> claim: Over-training was accompanied by alterations in the lipid profile, which appeared to be the consequence of over-training.
> claim: The negative effects of intensive training at a young age were outweighed by the many social, psychological and health benefits that a s...

### 640  [grade-school-math]
router: claude-instant-v1 ($0.0005, score 0.25)  |  oracle: code-llama-34b-instruct ($0.0003, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

### 715  [grade-school-math]
router: claude-instant-v1 ($0.0006, score 0.50)  |  oracle: code-llama-34b-instruct ($0.0004, score 0.75)

> TASK: Solve the following grade school math problem and provide a numerical answer.
> The following are examples of grade school math problems and answers:
> Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
> Answer: 6
> 
> Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 
> Answer: 5
> 
> Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
> Answe...

## zoib | overspend | oracle=claude-instant-v1 (n=14)

### 2456  [mmlu-high-school-chemistry]
router: gpt-4-1106-preview ($0.0016, score 1.00)  |  oracle: claude-instant-v1 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> A solution of sulfurous acid, H2SO3, is present in an aqueous solution. Which of the following represents the concentrations of three different ions in solution?
> A) [SO32-] > [HSO3-] > [H2SO3]
> B) [SO32-] = [HSO3-] > [H2SO3]
> C) [H2SO3] > [HSO3-] > [SO32-]
> D) [H2SO3] > [HSO3-] = [SO32-]
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 2385  [mmlu-elementary-mathematics]
router: gpt-4-1106-preview ($0.0008, score 1.00)  |  oracle: claude-instant-v1 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Find the quotient of 16.1 ÷ 0.14.
> A) 1.15
> B) 0.0115
> C) 11.5
> D) 115
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

### 2364  [mmlu-elementary-mathematics]
router: gpt-4-1106-preview ($0.0009, score 1.00)  |  oracle: claude-instant-v1 ($0.0001, score 1.00)

> Please answer with the letter of the correct answer.
> 
> Write 1 and 1 over 11 as a whole or mixed number in simplest form.
> A) 11 over 11
> B) 1.1 over 11
> C) 11 over 12
> D) 12 over 11
> Print only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:

## zoib | overspend | oracle=claude-v2 (n=1)

### 2114  [mbpp]
router: gpt-4-1106-preview ($0.0225, score 1.00)  |  oracle: claude-v2 ($0.0078, score 1.00)

> Write a function to find the count of all binary sequences of length 2n such that sum of first n bits is same as sum of last n bits.
