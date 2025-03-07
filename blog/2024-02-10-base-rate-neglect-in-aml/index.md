---
title: "Base Rate Neglect in AML"
author: ""
date: "2024-02-10"
toc: true
categories: [Article, Anti Money Laundering, Probability, R]
description: " A Probabilistic approach to AML Detection"
lastmod: "2024-02-10T12:38:54-05:00"
draft: false
---

For decades, false positives have plagued Anti Money Laundering (AML) detection efforts, creating an industry-wide challenge. Stories abound of Investigation teams creaking under the ever increasing burden of noisy alerts. Addressing this ever growing volume of alerts is an industry unto itself.

In this post, I will argue that the problem may not be with the alerts or detection system itself but how we interpret what they mean and our failure to view them them through the right lens of probability.

One of the classic examples that highlight our inability to assess probabilities accurately is the medical test paradox.(see footnote for 3B1B's excellent video)

## The Medical Test Paradox [^1]

[^1]: https://www.youtube.com/watch?v=lG4VkPoG3ko

Assume that you take a medical test with a sensitivity of 0.99 and a specificity of 0.9 . If the test comes back as positive, what is the probability you have the disease ?

**Sensitivity** or **True Positive rate** is the likelihood that the medical test (T) will return a positive result (+) if a patient has the disease(D).

$$ P(T = + \mid D) = 0.99 $$ **Specificity** or **True Negative Rate** is the likelihood that the test will return a negative result if a patient does not have a disease.

$$ P(T = - \mid \neg D) = 0.9 $$

The complement of the True Negative Rate or Specificity is the False Positive rate.

$$  P(T = +  \mid \neg D) = 1 - P(T = - \mid \neg D) = 1 - 0.9 = 0.1 $$ The True Positive Rate and False Positive Rates represent **Sampling probabilities**. Given a hypothesis or condition (e.g. the patient has a disease), it tells us how likely we are to observe a specific outcome (e.g. medical test is positive ).

The value we need to compute is $P(\frac{D}{T = +})$ . This is also known as the Positive Predictive Value. This is an **Inferential Probability**. Given an observation (medical test is positive), it tells how likely a hypothesis (patient has the disease) is.

This just requires a simple application of Bayes rule.

$$ P(D \mid T = +) =  \frac {P(T = + \mid D) \times P(D)}{P(T = +)}  $$

Note the denominator in the above expression simply evaluates the total probability of the test being positive which is give by.

$$ P(T = +) =  P(T = + \mid D) \times P(D) + P(T = + \mid \neg D) \times P( \neg D) $$

This simply accounts for the probability that the test can be positive when the patient has the disease or is free of the disease.

Note also that $P(T = + \mid \neg D) = 1 - P(T = - \mid \neg D)$. We already know the value of the latter term (True Negative Rate).

We already have most of the values we need to compute the required probability. Plugging them below, we get

$$ P(D \mid T = +) =  \frac { 0.99 \times P(D)}{ 0.99 \times P(D) + ( 1- 0.9) \times P( \neg D)}  $$

This leaves us with two terms we need to determine. $P(D)$ is the prior probability you will assign to someone having a disease.

Now if this is an extremely rare disease or if you are looking at a patient who is very unlikely to have this disease based on medical history or other demographic and socio-economic factors, you would assign a low prior probability.

For example, the prior for Bob having Covid when he has been sheltering in place for the last 1 month is really low. Say $P(D) = 0.01$.

On the other hand, Alice has been travelling for the last month without wearing a mask, the prior for Alice would be quite high. Say $P(D) = 0.25$

With this information, we can calculate the posterior probability that a patient has the disease given a positive test.

This simple function implements the above formula.

``` r
posterior_prob <- function(prior,sensitivity,specificity){
  
  post <- (sensitivity * prior)/(sensitivity*prior + (1- specificity)*(1 - prior))
  return(post)
  
}
```

``` r
cat("Probability of Bob having Covid is:",round(posterior_prob(0.01,0.99,0.9),2))
```

```         
## Probability of Bob having Covid is: 0.09
```

``` r
cat("Probability of Alice having Covid is:",round(posterior_prob(0.25,0.99,0.9),2))
```

```         
## Probability of Alice having Covid is: 0.77
```

As you can see, even though the medical test has an ostensibly high accuracy,the likelihood that a patient testing positive has Covid is not nearly as high as one would expect.The priors also make a significant difference to the eventual diagnosis.

I contend that AML models and scenarios are equivalent to medical tests. Just as medical tests tell us whether a patient has a disease, an AML model is supposed to tell us whether a customer is a money launderer by creating an alert or a case. Just as with medical tests, how we interpret the alert or case is important.

### Posterior Odds with Bayes Factors

An equivalent and perhaps more intuitive way of solving this problem is with odds rather than probabilities.To do this

1)  Express the prior as odds. Given a probability $p$, the odds are given by $\frac{p}{1-p}$

``` r
prob_to_odds <- function(p) p/(1-p)
```

For Bob, the prior odds would be $\frac{0.01}{0.99}$.

``` r
prob_to_odds(0.01)
```

```         
## [1] 0.01010101
```

For Alice, this would be $\frac{0.25}{0.75}$

``` r
prob_to_odds(0.25)
```

```         
## [1] 0.3333333
```

2)  Calculate the Bayes Factor.

This is the ratio of the sensitivity(TPR) of the test to the FPR. $$ BF = \frac{Sensitivity}{FPR} =  \frac{P(T = + \mid D)}{P(T = + \mid \neg D)}  $$ The simple function below computes the Bayes Factor.

``` r
bf<- function(sensitivity,specificity){
  return(sensitivity/(1 - specificity))
}
```

The BF for the test in question is given by

``` r
round(bf(0.99,0.9),2)
```

```         
## [1] 9.9
```

3)  Calculate the posterior odds by multiplying the prior odds with the Bayes Factor.

``` r
cat(" The odds of Bob having Covid is:",round(prob_to_odds(0.01)*bf(0.99,0.9),2))
```

```         
##  The odds of Bob having Covid is: 0.1
```

``` r
cat(" The odds of Alice having Covid is:",round(prob_to_odds(0.25)*bf(0.99,0.9),2))
```

```         
##  The odds of Alice having Covid is: 3.3
```

Given odds of $o$, the probability $p$ is given by $\frac{o}{1+o}$.

We can confirm the odds compute here are same as the probabilities computed above.

``` r
odds_to_prob <- function(o) return(o/(1+o))
```

``` r
cat("Probability of Bob having Covid is:",round(odds_to_prob(0.1),2))
```

```         
## Probability of Bob having Covid is: 0.09
```

``` r
cat("Probability of Alice having Covid is:",round(odds_to_prob(3.3),2))
```

```         
## Probability of Alice having Covid is: 0.77
```

Just as the medical test paradox sheds light on common misinterpretations of probability, the Prosecutor's Fallacy illustrates these challenges in a legal context, further emphasizing the need for probabilistic thinking in AML.

## The Prosecutor's Fallacy

It is not just doctors, but also lawyers, juries and judges who have sometimes failed to interpret probabilities correctly.

In the 1968 case, People vs Collins , Malcolm and Janet Collins, an interracial couple, were arrested for robbery as their characteristics matched those given by an eye witness.

The prosecution made the argument that the chances of the couple just happening to be at the crime scene by pure chance was extremely low.

The probability of observing each of the couple's characteristics were given as follows.

| Observation                     | Probability |
|:--------------------------------|:------------|
| Man with mustache               | 1/4         |
| Woman with blonde hair          | 1/3         |
| Woman with pony tail            | 1/10        |
| African American man with beard | 1/10        |
| Interracial couple in a car     | 1/1000      |
| Partly yellow car               | 1/10        |

: Table 1: Probability of seeing a match with characteristics seen by Eye witness

Consider the hypothesis (H) that the couple is innocent. Under this hypothesis, the probability (D) that an eyewitness would see a couple with the specified characteristics is astonishingly low, calculated as 1 in 12 million—a product of the individual probabilities detailed above. Note this assumes these characteristics are independent which is almost certainly not true and actually underestimates the probability.

The couple were initially convicted based on this argument.If the likelihood of them being innocent is so small, then they ought to be guilty.

The problem is that this probability gives the probability of observing a couple with the given characteristics under the hypothesis that the couple were innocent.

This is P(D\|H), the **sampling probability**.

However, what we are really interested in is the probability that the hypothesis is true given we observed a couple with the aforementioned characteristics - P(H\|D) or the inferential probability.

This can again be calculated using Bayes rule as follows

$$ P(H \mid D) =  \frac {P(D\mid H) \times P(H)}{P(D)}  $$ where

$$ P(D) = P( D \mid H) \times P(H) + P( D \mid \neg H) \times P( \neg H) $$

Now we have to determine priors - $P(H)$ , the prior probability that the couple is innocent and $P( \neg H)$ , the prior probability that the couple is guilty. Note these are to determined before we observe any evidence i.e. the eye witness account.

Given there were some 5 million couples in the Los Angeles area where the crime was committed, and any one of these couples could have committed a crime, a reasonable prior is

$$ P(\neg H) = \frac{1}{5,000,000}  $$

This implies

$$ P(H) = \frac{4,999,999}{5,000,000}$$

The other term we need to determine is $P(\frac{D}{\neg H})$, This is the probability of observing the stated characteristics in table 1, if the couple were guilty. This will be **1**.

Plugging these values in, the posterior probability that the couple is innocent is given by

$$ P(H \mid D) =  \frac {\frac{1}{12,000,000} \times \frac{4,999,999}{5,000,000}}{ \frac{1}{12,000,000} \times \frac{4,999,999}{5,000,000} + 1 \times \frac{1}{5,000,000}} = 0.294 $$

The posterior probability hat the couple is innocent is actually close to 30% rather 1 in 12,000,000. This certainly does not prove anything beyond a reasonable doubt.

You will be glad to know the Collinses' were acquitted by the California Supreme Court on appeal where the presiding judge commented :

> Undoubtedly the jurors were unduly impressed by the mystique of the mathematical demonstration but were unable to assess its relevancy or value

## Why this matters in AML

My contention is that AML practitioners also fall victim to base rate fallacy. In place of intervening medically when not needed, or convicting innocent people, AML practitioners tend to investigate cases or file SARs when there isn't compelling evidence to do so.

Much like in the case against the Collinses, we tend to make decisions based on sampling probabilities rather than inferential probabilities. The chances of all these models triggering an alert on a regular customer is really low, so if they do alert, the customer must be worth investigating.

Consider how transaction monitoring is typically done at most institutions. The transaction activity of a customer is monitored for patterns like Change in Behavior or Rapid Movement or Rapid Flow Through of Funds. If a customer triggers alerts on some combination of these scenarios, a case is created.

Some institutions are so conservative that a case is created if an alert is triggered for just one of these scenarios. For others, a case is created only if some combination of these scenarios triggers an alert.

Let us assume under regime 1. we create and investigate a case when any one of these scenarios triggers an alert. Under regime 2 on the other hand, we create a case only when all three scenarios trigger an alert.

Now each of these scenarios are equivalent to a medical test. When a medical test returns a positive result, a physician can choose to intervene. Similarly, when a scenario triggers an alert, an investigator can choose to intervene and investigate the customer. The key question is how much evidence is sufficient to justify this intervention.

Let us assume both institutions use three different scenarios or models - Scenario A (Change in Behavior) ,Scenario B (Rapid Flow Through of Funds) , Scenario C (High Risk CounterParty), with the goal of detecting Shell Accounts that might be operating through the institution.

The hypothesis here is that a shell company will demonstrate these underlying patterns.

I will also assume the scenarios alert independent of each other and make the following assumptions for each scenario.

Probability of a customer showing target pattern given customer is a shell company = 0.99 <br> Probability of a customer showing target pattern given customer is not a shell company = 0.01 <br> Probability a scenario alerts in the presence of the pattern = 0.9 <br> Probability a scenario alerts in the absence of the target pattern: 0.05

For each of the models, the performance of the scenario is as follows

Sensitivity[^2] is 0.8915 (0.9 x 0.99 + 0.05x(1 - 0.99)). If the customer is truly a criminal, this is the probability that a model triggers and an alert.

[^2]: P(S=Alert \| Cust = Criminal) = P(S=Alert \|(Cust shows Pattern) x P(Cust Shows Pattern \|Cust = Criminal) + P(S =Alert \|Cust shows no pattern) x P(Cust shows no pattern \| Cust = Criminal)

Specificity[^3] is 0.9415 (1 - 0.9 x 0.01 + 0.05\*(1-0.01)). If the customer is a regular citizen, this is the probability that the model does not alert.

[^3]: 1 - P(S=Alert \| Cust = Citizen) = 1 - P(S=Alert \|(Cust shows Pattern) x P(Cust Shows Pattern \|Cust = Citizen) + P(S =Alert \|Cust shows no pattern) x P(Cust shows no pattern \| Cust = Citizen)

| Scenario   | Sensitivity | Specificity |
|:-----------|------------:|------------:|
| Scenario A |      0.8915 |      0.9415 |
| Scenario B |      0.8915 |      0.9415 |
| Scenario C |      0.8915 |      0.9415 |

: Table 2: Accuracy of scenarios

Note this is the performance you would see only with a high quality scenario.

Now,a critical question. What is the prior probability that a customer at either Financial Institution (FI) is a bad actor or criminal.

Let us assume FI 1 is a regional bank in the US mid west. Here the prior probability of any given customer being a money launderer is very small. Say 1/1,000,000.

Let us assume FI 2 is an international bank offering services in tax havens and high risk jurisdictions, here the prior would be an order of magnitude bigger. Say 1/100,000

### FI 1 under regime 1

Here, we have an FI with a low risk customer base which also has a very conservative approach to reviewing cases.

The posterior probability of an alerted customer being truly suspicious is truly minuscule.

``` r
#Prior odds x Bayes factor for one scenario
odds_to_prob(prob_to_odds(1/1000000)* bf(0.8915,0.9415))
```

```         
## [1] 1.52391e-05
```

### FI 2 under regime 1

What about the FI with a higher risk customer base ?

Even here, a conservative approach to investigating cases leads to an extremely low posterior probability of a case being truly effective, albeit three orders of magnitude higher than for FI 1.

``` r
#Prior odds x Bayes factor for one scenario
odds_to_prob(prob_to_odds(1/1000)* bf(0.8915,0.9415))
```

```         
## [1] 0.01502537
```

### FI 1 under regime 2

What about FI 1 with a less conservative approach to creating cases ? It gets only marginally better.

``` r
#Prior odds x Bayes Factor for three independent scenarios
odds_to_prob(prob_to_odds(1/1000000)* bf(0.8915,0.9415)^3)
```

```         
## [1] 0.003526652
```

### FI 2 under regime 2

What about FI 2 with a less conservative approach to creating cases ?

``` r
#Prior odds x Bayes Factor for three independent scenarios
odds_to_prob(prob_to_odds(1/1000)* bf(0.8915,0.9415)^3)
```

```         
## [1] 0.7798652
```

Only in this case does, even such a set of high quality scenarios yields cases that are truly likely to result in a SAR.

## Takeaways

Here is what I think we should do differently to improve the efficiency of AML Detection programs.

1.  Instead of creating cases based on binary rules or models, models or rules should output probabilities that represent the posterior probability of a case being truly effective.

    Institutions can then choose to review cases only when the posterior probability is greater than some threshold which they can choose based on their risk tolerance. If they choose to review cases with a very low posterior probability as FI 1 chooses to do, then they should expect a lot of false positives.

    Alternately,institutions can wait until more evidence accumulates before investigating a case. This can lead to creation of higher quality cases

2.  Create better segmentation. A good risk based segmentation system can create segments where the prior probability is much higher which in turn can lead to higher quality cases.

3.  Create better features. This is a no brainer. Features that can more accurately detect the underlying patterns of interest will lead to improved detection. Deriving features from graph representations of underlying transaction data or time series data are promising avenues to consider.

4.  Assign importance to alerts based on how discriminating the underlying patterns are. If a scenario monitors for highly generic patterns that can manifest for both regular customers and bad actors, alerts will be very noisy by design. The weight given to alerts from such scenarios should be low given how noisy they can be.

5.  Create richer cases by collecting as many signals as you can. Transaction data can only tell you so much. Incorporate signals from external data sources such as negative news stories or ICIJ data sets to improve the quality of cases.

## References and Resources

1)  [Bernoulli's Fallacy: Statistical Illogic and the Crisis of Modern Science](https://www.amazon.com/Bernoullis-Fallacy-Statistical-Illogic-Science/dp/0231199945) <br>
