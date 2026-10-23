# Operationalizing the Persona Selection Model

**TL;DR** I attempt to extract testable predictions out of the persona selection model. I first elicit a set of persona descriptions from a base pre-trained model (Olmo3).
I then fit the responses of an assistant in the base model as a mixture of responses from these different personas, which works quite well (achieving within 0.006 nats/token of the base model itself).
I then attempt to fit the post-trained Instruct model with this same mixture, but a large gap remains (0.82 nats / token). 
Overall qualitatively this confirms some aspects of the PSM: base model is reasonably approximated by a mixture of personas, and personas can be extracted from the base model and be used to understand the post-trained model behavior.
But quantitatively the large unexplained variance of the Instruct model suggests post-training is more complicated than mere selection / routing. 
Feedback requested!

## Introduction

The Persona Selection Model (PSM) is the dominant mental model many alignment researchers are using to make sense of model behavior and strange generalization effects like Emergent Misalignment. 
There is a lot of very interesting discussion on Lesswrong is about this model, to what extent it holds and where it might be lacking. 
Unfortunately the PSM is a quite 'squishy' model, that is ("difficult to extract important predictions from") [https://www.lesswrong.com/posts/csRby7mZgjL5jCoLL/thoughts-on-the-persona-selection-model].
I set out in this project to try write down explicit formulations of the PSM, and extract testable predictions from them, and see where things break.
This is attempting some sort of science of personas. 

I am particularly focused on the relationship between base and post-trained models. How much of the behavior of the behavior of the post-trained model can be understood from interrogating the base model?
In the PSM the answer should be 'a lot' (or in its strong form 'all of it'?). 
The devil being in how to properly extract the relevant components from the base model and relate them to the post-trained model.
I attempt a first simplistic approach here, in which I elicit a set of personas from the base model, and then fit Assistant responses from the base and post-trained models as a mixture of these 
personas.
The fitted persona-weights tell us how much each persona archetype is contributing to the overall behavior of the model. 
The gap between the likelihood of model actually generating the responses and that of the best-fit mixture tells us how well this mixture actually describes the assistant behavior. 
For the base model the answer is 'pretty well' but for the post-trained model a significant gap remains. 
This suggests that post-training is indeed doing something more complicated than this naive formulations of the PSM can account for. 

In a companion post ([Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss](my_writeup_base_model_likelihoods.md)) I show a way in which base models can still be quite a useful tool for understanding post-trained models.

I will also say from the outset that I am a physicist pivoting into AI safety and this is my first (solo-)project. So I would sincerely like feedback! 
I don't want to fall into the trap of the [relevant xkcd](https://www.explainxkcd.com/wiki/index.php/793:_Physicists).

## The Persona Mixture Model

One relatively simple operationalization of the PSM is a mixture model, where we describe the probability distribution of responses from the model as a mixture of different 
personas along with a function that determines which persona to select in the given context. This leads to the follow form:

$$
P(x \mid c) = \sum_s P_R(s \mid c) P_s(x \mid c)
$$

That is, the probabilty of observing a response from the model $x$  in a certain context $c$ is a mixture model over different personas $s$. 
$P_R$ is a 'routing function' that selects how likely a certain persona is given the context $c$, and $P_s$ are a basis of different personas.

A strong version of the PSM would say that the persona basis $P_s$ is fixed by the pre-training, and post-training attempts to shape the routing function
to always point to the desired persona (call it HHH) regardless of context; ie $P_R(s = HHH) = 1, $P_R(s != HHH) = 0$. 

To use this mixture model in practice, the key question then is how to determine $P_s$, the probability distribution of a specified persona?
I attempt to elicit this from the base model, by using transcripts headers of the form:
*"Below are a series of dialogues between various people and an AI assistant. The AI assistant is [one paragraph describing the specific assistant persona]"*, followed by `User: [query] Assistant:`
Appending an assistant repsonse after this header and evaluating its likelihood is what I use as a proxy for $P_s$. 

To simplify the mixture even further, I also make the routing context independent : $P_R (s \mid c) = w_s$.
Which means the model has some fixed probability $w_s$ to employ a given persona regardless of the context. 
The $w_s$ are just mixture coefficients and can be determined empirically by fitting our mixture model to a set of responses sampled from a given model.


**Models** I test this approach using the `Olmo-3-7B` family of models, using `Olmo-3-1025-7B` as the base and `Olmo-3-7B-Instruct` as the post-trained model. 
I also test on Qwen2.5-7B and Qwen2.5-7B-Instruct for some cross-family checks. 


**Personas.** I started with six hand-written one-paragraph descriptions (Askell's original HHH assistant description, an evil one, a plain one, a sycophant, a formal one, a dismissive human).
I sampled an additional 80 more from the base model itself by asking it to continue "Character description: The assistant…". 
I thought this might be an interesting way to see what personas the base model itself thinks are the most likely a priori. (96 total were generated but filtered down to 80 due to duplicates or malformed repsonses). 
Of those 80, 39% describe a friendly helper, 34% a named human, 26% a sarcastic character, 12% a blunt one, 4% a sinister one; none an explicitly malicious assistant. 
Expanding this basis and finding more clever ways to elicit a full persona basis from tbe base model is an obvious area for improvement. 



**Data.** 300 first-person temptation dilemmas generated by an LLM ("The cashier gave me \$20 too much change. What should I do?"). 
This style of query was chosen to elicit different behaviors from the personas. 
"The assistant speaks in a casual, conversational tone" is appended to the generic and to every persona header alike, so that the primary differences in the personas are content rather than manner of speaking.
When sampling from the Instruct model, each user query ends with the statement "Answer in two or three sentences of plain text, in a casual, conversational tone".
Standardizing the register in this way was found to improve the quality of the mixture fit. 

**Base model assistant** To sample responses from the base model, to be fit with this mixture, I apply a header saying only that the assistant "has a well-defined character of its own, not known a priori". 2,400 samples; 2% are placeholders or meta-commentary and are removed.

**Fit.** Score every response under every persona header. Choose the weights $w_s$ that maximise the likelihood of the samples. 
Fit on half the questions, evaluate on the other half. 
I tested this method on synthetic mixtures of responses sampled from specific personas, and it recovered the correct weights to within 2-3% and the fitted mixture model had a negligible likelihood gap with respect to the true distribution. 

**Results.** 
The main metric to report is the log P / token achieved by the mixture fit. 
The difference between the mixture fit and the log P / token of the generating model itself (the KL) tells us how close the mixture model is to describing the generating model.

On the base model, the base model with the generic header has an average log P / token of −1.612 ± 0.012:  

| basis \ header | plain header | header + shared casual clause |
|---|---|---|
| six hand-written personas | −1.632 ± 0.014 (gap 0.031 ± 0.002) | −1.603 ± 0.014 (gap 0.014 ± 0.001) |
| all 86 (hand-written + elicited) | −1.613 ± 0.014 (gap 0.011 ± 0.001) | **−1.594 ± 0.014 (gap 0.006 ± 0.001)** |

| scored under | plain header | header + shared casual clause |
|---|---|---|
| sampler itself (generic header) | -1.601 ± 0.013 | -1.589 ± 0.014 |
| mixture of six hand-written personas | -1.632 ± 0.014 (gap 0.031 ± 0.002) | -1.603 ± 0.014 (gap 0.014 ± 0.001) |
| mixture of all 86 personas | -1.613 ± 0.014 (gap 0.011 ± 0.001) | -1.594 ± 0.014 (gap 0.006 ± 0.001) |
| best single persona | -1.629 ± 0.014 (gap 0.028 ± 0.002; easygoing chatty AI) | -1.604 ± 0.014 (gap 0.016 ± 0.001; plain assistant) |

![Greedy elbow: hold-out log P per token of the base assistant's answers vs number of personas](results/phase1/elbow_logp.png)

Expanding the basis specifying the and register each halve the log P gap. Both can probably be taken further to reduce the gap even more. 
The best fit puts 24% on a "plain assistant" ("the assistant answers the users questions and responds neutrally"), 16% on an "opinionated hedger", 14% on a "kind but sarcastic friend with profanity".
Overall the mixture seems like a pretty good fit. The 'plain assistant' persona being dominant makes sense, but I was somewhat surprised the 'opinionated hedger' is so high.
I was honestly expecting more variation in the assistant behavior (ie more personas being 'active'), as if you randomly sampled across stories you would certainly find more varied characters.
I would guess some amount of chat style documents are in the pre-train of the model, which is limited the variation in this context.
It might be interesting to try something similar in a 'story' framing and see how the mixture changes. 

Fitting the instruct model:

| Instruct's answers scored under | log P / token | gap to Instruct, nats / token |
|---|---|---|
| Instruct itself (exact continuation) | −0.898 ± 0.009 | 0 |
| base, generic header | −1.758 ± 0.012 | 0.860 |
| base, mixture of 86 personas | −1.722 ± 0.012 | 0.824 |
| base, best single persona ("college student") | −1.731 ± 0.012 | 0.833 |

Looking at the largest weight personas, the HHH persona is given 38%, which makes sense, but interestingly the dominant persona (39%) is actually a "college student", who is described as:

"The assistant is a young college student in his early 20s. He has short black hair and hazel eyes, sometimes glasses. He has an easy-going demeanor, a bit quirky, but sincere and friendly. Though he's smart and eager to succeed, he's not a showoff, instead being humble and modest. He has an inquisitive mind and a tendency to see the world with a healthy skepticism."
 
This might have to do with the Instruct assistant being well spoken, rather than the behavior. Interestingly, it seems related to recent evidence from Story Imprinting which showed the model did associate the Assistant behavior with a student from an elite college. 

But overall, the mixture is a much worse fit to the Instruct model, a large gap in log P / token remains which does not close with more personas.
The mixture provides a slightly better fit than the single best persona, and somewhat better than the base model with a generic header, 
so there is some 'signal' being fit, but not much.

Overall the Instruct model has much lower entropy than any of the personas elicited from the base model. 
Evaluating the probability of responses generated from an base-model persona under that same model is keeps things in the log P / token ~ -1.6 range, so it is no mystery why the base model personas cannot match the likelihood of the Instruct model itself.
However, attempting a version of the mixture fit with sharpened versions of the base-model personas (evaluating the likelihood of the top personas  with a temperature T < 1.0) only makes the fit worse.
So I think its both that the span of personas used are not a good basis for the Instruct model, and that the Instruct model persona is much sharper than anything from the base model.

Before moving on, one might wonder whether I have just not found the right context / persona to describe the behavior of the Instruct model.
A stronger version of the PSM states that "post-training is just conditioning".
This can also be operationalized. 
One can attempt to optimize the condition (context) that to maximize the likelihood of our sampled response from a base model. 
This has recently been used on Instruct models to reconstruct the teacher's hidden system prompts in a subliminal learning setup (https://arxiv.org/abs/2609.16927).
Optimizing a soft-context on the Instruct model to maximize the likelihood of a set of responses generated with a hidden system prompt works pretty well,
achieving a likelihood within 0.04 nats/token of that of the generating model (-0.89). 
Performing the same procedure on the base model, to maxmize the likelihood of responses taken from the Instruct model, does better than the mixture or a generic header, achieving 
-1.37 nats/token, but this is clearly still well short of the Instruct model itself. 
It plateaued around this point regardless of context length, learning rate, initialisation, or data size.

Though the persona basis and setup can clearly be improved, these results make me quite skeptical one could ever fully describe the probability distribution of the 
post-trained model as a mixture of these context-elicited base model personas. 
I think this means that if the PSM is to hold rigorously, whatever a 'persona' is cannot be something fully specified by context. 
I think the 'base units' are closer to tropes, associations between behaviors in different contexts, that the model picked up on during pre-training, rather than fully formed personas or characters. 

However, these base model elicited personas can still cary a useful signal for understanding the behavior of post-trained models, which I demonstrate in a [companion post](my_writeup_base_model_likelihoods.md).
