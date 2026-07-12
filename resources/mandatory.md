# Qaulitative results / Reasoning
1. This assignment is designed to evaluate how you approach an applied research problem end- to-end — from understanding the data to making *design decisions, implementing a solution, and communicating your findings clearly*.
2. This is a deliberately ill-posed problem: SAR carries no direct colour or spectral information, so a given SAR input is consistent with many plausible EO outputs. We are interested in how you
reason about that difficulty.
3. We are not only evaluating your final metric scores. We are equally interested in how you think, what trade-offs you identify, and how clearly you communicate your reasoning. A well- reasoned approach with modest numbers is more valuable to us than strong numbers with no justification.
4. You are responsible for constructing your own train / validation split from the permitted data. Document your split strategy. Be deliberate about it: adjacent satellite patches can be near- duplicates, so a naïve random split may leak information between train and validation and inflate your reported numbers. Consider splitting by scene, region, or season.
5. SAR and optical modalities differ in dynamic range, noise profile, and statistics. SAR amplitude is typically log/dB-scaled before use; optical RGB requires its own normalisation. How you preprocess and normalise each modality is a design decision and should be stated in your report.
6. 




# proof
1. metrics calculated on val
2. 
3. ablation
4. 


1. should not use dataset outside of mentioned ones
2. controlled ablation
3. log training and validation loss per epoch, save the loss-curve plot, and persist the raw loss values (CSV or JSON)
4. stick with perpetual metrics rather than pixel-level and discuss the gap between your pixel and perceptual numbers
5. include qualitative results and comment on error profile