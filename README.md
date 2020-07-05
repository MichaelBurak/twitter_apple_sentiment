


> Written with [StackEdit](https://stackedit.io/).
> ![enter image description here](https://i.imgur.com/P7JnVmp.jpg)
> # **"Tells” in Tweets: Developing a reliable Twitter sentiment analysis classifier for tech companies**
> 
## Business value: 
Being able to reliably classify tweets allows for a jumping off point for further analysis and modeling, as well as more ability to choose which tweets to examine for other use cases.

## Business use case: 
By identifying the negative and positive sentiments about Apple and Google, a tech business from the outside would be able to better understand the strategies, successes and blunders in the eye of the public for two tech giants. This would allow them to better their business by avoiding past mistakes or customer paint points and bouncing off of the successes of these companies.

## Evaluation: 

Cohen's Kappa score has been chosen as the metric of evaluation, especially fit for the imbalanced dataset and measuring the reliability of the classifier, how precise it is compared to being rated by chance. A robust and reliable classifier is needed for this business case due to how noisy and complicated twitter data can be.

# Data Preprocessing and EDA

- The data consisted originally of three features:
	- The text of a given tweet written during the period of SXSW in Austin, TX, ostensibly regarding Apple or Google or related products/service.
	- The product/service/company the tweet was directed at.
	- The sentiment of the tweet, to be collapsed down to Negative, Neutral and Positive, later recoded to 0, 1, 2 as numerical values for analysis.
	
### Null values: First, a large number of null values were found in the data in one column and removed:
This feature is dropped to focus on the text itself and later, statistics regarding the text, to make a more generalizable classifier than including information about the direction of the tweet, as well as a significant amount of null values compromising this feature.
![Out of a sample of 1000, less than 400 of the column dictating what product or company the tweet was directed at were non-null](https://i.imgur.com/1VqosYA.png)

### Columns renamed for easier reference and sentiment mapped to numerical values:

    df = df.rename(columns={"tweet_text": "text", "is_there_an_emotion_directed_at_a_brand_or_product":"sentiment"})
    ...
    df.sentiment = df.sentiment.map({'Positive emotion': 2, 'Negative emotion': 0, 'No emotion toward brand or product': 1, "I can't tell":1})
 
## A core issue: severe class imbalance
![enter image description here](https://i.imgur.com/FWD0xVi.png)

## Text cleaning and visuals
- The text was then preprocessed mildly including to clean it of some html tags, punctuation marks, and other characters that would hinder analysis.
- Analysis was performed on the cleaned text after features were created out of the length of a tweet, words in a tweet, and amounts of hashtags, capital words, exclamation/question marks, the mean length of a word in a tweet, the count of unique words in a tweet, and the percentage of unique words in a tweet.
	- Pairplot of all numerical variables
![enter image description here](https://i.imgur.com/yW5pyLZ.jpg)
- Correlation matrix(diagonal masked)
![enter image description here](https://i.imgur.com/8xN58ZA.png)
- Tweet_len was dropped as a reasonably redundant feature captured in and correlated with other features.
- Tokenizing, stemming, lemmatization were performed.
	- A wordcloud before:
	 ![enter image description here](https://i.imgur.com/ehwvaff.png)
	- Wordcloud after:
	![enter image description here](https://i.imgur.com/r6X3VvE.png)
- TSNE was performed on various aspects of the cleaned text:
	- For more clustering and details, see the notebook.![enter image description here](https://i.imgur.com/w1n4uRP.png)
- Frequent bigrams were calculated and plotted out, including on a graph structure, showing definite trends in the data around Google's social network rumored to be called "Circles", the iPad 2, and Apple's popup store during SXSW 2011: 
![enter image description here](https://i.imgur.com/wV1R9hE.png)
# Modeling
- Extensive early testing showed these LinearSVC with balanced class weights and ComplementNB to be the best compromise between performance and speed on this dataset without using tree-based methods(author's note: I utilized trees extensively in my last work, so I am forgoing them this time.)
- TF-IDF vectorization capturing unigrams and bigrams was performed, as well as a stratified train-test-split to try to mitigate class imbalance.
- Initial modeling was poor, Cohen's Kappa of Linear SVC was 0.018 and the Cohen's Kappa of ComplementNB was 0, but MinMaxScaler raised these to  0.323 and 0.346.

## On comparing models: 
- Linear SVC seemed to perform well on text classification on this dataset, along with ComplementNB right behind(.345 kappa). However, the number of parameters and tuning process in high dimensional feature space of Linear SVC limits its usefulness compared to the much faster Complement Naive Bayes without dimensionality reduction in place.

## Iterative SelectKBest:
- An iterative version of SelectKBest to derive an optimal number of features with a grid searching of ComplementNB was performed, selecting 30000 features out of a previous 50000+ and achieving a Cohen's Kappa of .509.
	- This is very impressive performance by Complement NB with tuning all things considered, tuned LinearSVC hds performed well, however with the computational load of tuning LinearSVC, ComplementNB has similar performance with more speed. It is worth noting SVMs scale better to larger text in the literature (https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf), but with more moving parts than the Bayseian classifiers.

# Interpretation:
- As a probabilistic classifier, I focused on the most predictive features in interpreting and evaluating the model.
![enter image description here](https://i.imgur.com/Gjn3nNq.png)

- The iPad and Apple seem to win the day, contextualizing and being contextualized by the coefficients unigrams and bigrams predicting tweets about it. 
- It looks like while they occurr frequently, with the iPad 2 showing up most of all, the model did not find key bigrams to be extremely predictive, likely due to inverting the term document frequency and their high rate of occurrence.

## Undersampling - an aside:
 - No undersampling techniques seem to benefit performance of the ComplementNB model after reducing feature space.
- A note about SMOTE: Despite good results in the last round of imbalanced classification I undertook as a project, SMOTE and its variants were not used as they synthetically create text that is not within the original dataset.

# Results and final interpretation:
## Model Results: 
- Achieved fair or moderately reliable score with ComplementNB in a reduced feature space through SelectKBest.
- This classifier struggled, even with weighting of classes, stratification, attempts at resapling through undersampling and more, with the heavy imbalance in the data. 
- All other classifiers struggled with the imbalance as well, the key impedement to performance in classification in this task. 
- Issues with the feasability of tuning LinearSVC and SGDClassifier led to using ComplementNB, similar performance without having to reduce the dimensionality further through something like PCA and sacrifice interpretability.
- Moving forward with this project and model, I would seek to better address imbalance.

## Model Interpretation: 
- Interpretation was made difficult by performance issues, especially on the negative class, and its low support, as well as the use of bigrams and the difference between the preprocessed and raw dataframe. 
- With that said, trends emerged in the EDA: the big showings of Apple and Google respectively with their popup store along with the iPad 2 and the rumored Circles social network contributed to positive ratings, while design critiques of the iPad 2 stood out in the negative class in EDA. 
- Likely due to the workings of TF-IDF, the model itself focused on other coefficients, contextualizing that Apple predominated with its iPad 2 showing with a large number of equally predictive coefficients.

# Business recommendations

## 1. Go Big: 
- The trends that stood out in the exploratory data analysis were big showings by Apple and Google during an important event. It is crucial to capitalize on hype and make bold moves during periods like SXSW: whether it be drumming up the launch of a new social network or creating a popup store. 

## 2. Data Integrity: 
- The analysis was heavily compromised by data issues. Data collection and labeling must be done with an eye towards class balance and adequate support. 

## 3. Focus on the tangible and reliable:
- The iPad 2 was a common theme among the text corresponding to the coefficients. Having an actual product as opposed to the rumors of Circles seems to have helped Apple considerably to win positive sentiment. Furthermore, utilize models such as ComplementNB that show how predictive a feature is to derive what are reliably predictive features.

# Future work: 
## Outlier detection approaches:
- Imbalanced learning can be treated as an outlier detection problem. 
- I would like to explore this through techniques such as Isolation Forest. 
- As well, I would look into adding additional features based off of the outlier and inlier values produced by those techniques to boost model performance.

## Multilabel classification
- This dataset could have been posed as a multilabel classification problem, by using the feature regarding direction of the tweet. 
- I would like to examine the various approaches to multilabel classification and see if they yielded better or different results and in what sense. 

# Sources: (Incomplete)
- https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
- https://www.datacamp.com/community/tutorials/feature-selection-python
- https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
- https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
