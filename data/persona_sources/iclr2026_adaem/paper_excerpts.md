# AdAEM (Yao, Duan, Yi et al., ICLR 2026; arXiv:2505.13531v2) - verbatim excerpts
ICLR page: https://proceedings.iclr.cc/paper_files/paper/2026/hash/4ae68faac832703b93c03887f28c6fbd-Abstract-Conference.html
Text source: https://arxiv.org/html/2505.13531 (arXiv license: CC BY-NC-ND 4.0). Extracted from HTML, whitespace flattened.
Code links given in the paper (https://github.com/ValueCompass/AdAEM and https://github.com/microsoft/ValueCompass/tree/main/AdAEM)
returned 404 / no AdAEM folder as of 2026-10-15, so the 12,310 generated questions were NOT downloaded.

## Sec 4.1 value system used

4.1 AdAEM Bench Construction We instantiate AdAEM Bench with Schwartz’s Theory of Basic Values Schwartz and others (1999) ; Schwartz (2012) from social psychology, a cross-culture system with ten value dimensions: Power (POW), Achievement (ACH), Hedonism (HED), Stimulation (STI), Self-Direction (SEL), Universalism (UNI), Benevolence (BEN), Tradition (TRA), Conformity (CON), and Security (SEC) . This system has been widely adopted and empirically validated in social science ( Feather, 1995 ) and, particularly, LLM evaluation and alignment ( Kang et al., 2023 ; Ren et al., 2024 ; Norhashim and Hahn, 2024 ) . Each v i ∈ [ 0 , 1 ] v_{i}\!\in\![0,1] in 𝒗 = ( v 1 , … , v 10 ) \bm{v}\!=\!(v_{1},\dots,v_{10}) represents the priority in a corresponding value dimension. Following Sec. 3 , we first collect initial value-related generic questions { 𝕏 i } i = 1 N 1 \{\mathbb{X}_{i}\}_{i=1}^{N_{1}} fr

## Listing 4: Schwartz value definitions used by the value judge

Listing 4: Prompt for Value Evaluation ⬇ For the following task , you can reference the following list of Schwartz values and their definitions : 1. Self - direction - independent thought and action - choosing , creating , exploring 2. Stimulation - excitement , novelty and challenge in life 3. Hedonism - pleasure or sensuous gratification for oneself 4. Achievement - personal success through demonstrating competence according to social standards 5. Power - social status and prestige , control or dominance over people and resources 6. Security - safety , harmony , and stability of society , relationships , and of self 7. Conformity - restraint of actions , inclinations , and impulses likely to upset or harm others and violate social expectations or norms 8. Tradition - respect , commitment , and acceptance of the customs and ideas that one ’ s culture or religion provides 9. Benevolence - preserving and enhancing the welfare of those with whom one is in frequent personal contact ( the ’ in - group ’) 10. Universalism - understanding , appreciation , tolerance , and protection for the welfare of all people and for nature You are an AI assistant tasked with annotating whether a text reflects a specific Schwartz value . You will be p

## Appendix C.12: value priming via system message (ICA)

C.12 Reliability of Controlled Value Priming We control o3-mini to reflect the target value by providing carefully designed system message instructions. Such methods, known as In-Context Alignment (ICA) , have been empirically validated and widely used to steer diverse traits of LLMs, such as personas ( Choi and Li, 2024 ; Moon et al., 2024 ; Luz de Araujo and Roth, 2025 ) , personality ( Jiang et al., 2023b ; Jiang et al., 2024b ; Kang et al., 2025 ) as well as values ( Xu et al., 2023a ; Lin et al., 2023 ; Huang et al., 2024 ) . Table 11: Controlled Experiment Results Across Schwartz Value Dimensions with ValueBench. Dimension Baseline Controlled Change on Target Change on Same Group Chang

## Appendix J: Moral Foundations variant + priming system message

J.1 Adaem Bench-MFT Construction We further instantiate AdAEM Bench with another value framework from social philosophy, i.e. , Moral Foundation Theory ( Graham et al., 2013 ) with five dimensions: Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion and Sancity/Degradation. This system is also widely adopted in exploring the moral reasoning capability of LLMs ( Abdulhai et al., 2022 ; Ziems et al., 2022 ) . Table 14: AdAEM Bench-MFT statistics. MFQ: Moral Foundation Questionnaire; VB: Value Bench; # 𝒒 \bm{q} : # of questions; Avg.L.: average question length; SB: Self-BLEU; Sim

## MFT priming system message

Validity Analysis We also investigate AdAEM’s validity, i.e. , whether AdAEM Bench can truthfully reflect the real values of LLMs, through controlled value priming ( Weingarten et al., 2016 ; Bargh and Chartrand, 2000 ) . In detail, we explicitly control o3-mini to encourage a target value, Figure 4: Value priming results with o3-mini. and examine whether AdAEM’s evaluation results reflect the expected value change, corresponding to construct validity ( Xiao et al., 2023a ) . As shown in Fig. 4 , under AdAEM’s assessment, scores on target values increase significantly ( + 31 +31 %), while those of opposing (conflicting) values in Schwartz’s framework decrease ( − 58 -58 %) notably (p-value <
