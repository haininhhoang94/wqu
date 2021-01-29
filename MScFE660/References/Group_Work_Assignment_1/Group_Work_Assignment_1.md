---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Group Work Assignment 1
Selected from Covid-19 Financial Crisis.

Develop a structed, concise paper that fully addresses the key themes of the 
chosen crisis and explore its social, economic, political and historical contexts.
You are required to have a clear introduction and conclusion, and to include
appropriate citations and references. Referencing guides for the Harvard or 
Chicago system are available online, or use the guidelines included in the
Student Resource Center (SRC)

The reader should come away from this paper with an understanding of:

a. the condition predecent that allowed the crisis to happen
b. the event or events which precipitated the acute phase of the crisis
c. the transmission mechanism which allowed the criss to spread
d. the factors which caused banks, industrial and service companies and or market
to fail
e. the resolution of the criss

### Submission Requirements
Submit the following for your assignment:
* A 2,000 - 3,000 report in PDF format, graphs, charts, results, from the 
developed code should be submitted to in the body of the paper. Source code may
be included in an appendix with comments sufficient to enable a reader to 
understand its operation. The comment on the source codeblock-full are excluded
from the word count


## Covid-19 crisis

### Social Impact
It is highly suggested by WHO that Covid-19 mostly spreads between people because of
direct or indirect contact with infected people. There are three transmitted causes
for Covid-19: Droplet Transmission, Contact Transmission, and Aerosol Transmission.
Droplet Transmission are occurred by droplet produced by contaminated patient, like
saliva and respiratory secretions. Contact Transmission is a type of indirect contact,
which mean that it occurs when a person touches a contaminated surfaces (contained
the virus from other patient), then touching their mouth or nose. The last transmitted
method, which is Aerosol Transmission, occurred when the surronding atmosphere
contains the virus at a radius of 6 feet, directly from the infected people, symptoms and
non-symtoms.

Consequently, most efficient method to tackle COVID-19 spreading is by Social Distancing.
By the definition of Centers for Disease Control and Prevention, Social Distancing is blocking
"physical contact", keep a distance of 6 feet from other people who are not in the same living
environment in both indoor and outdoor spaces.[2] Since physical contact is either not allowed 
or limited, people have been send to Work From Home or lay-off. The two industries that suffered
the most are airlines and petroleum.

```python
from IPython.display import Image
Image(filename='Data/Image/UnemploymentRate-2020.png') 
```

According to the United States Unemployment Rate from Trading Economics, from July
2019 to February 2020, the mean for unemployment rate is approximately 3.575%. However,
after the spreading of COVID-19 at the beginning of March 2020, the unemployment rate was sharply rose
to 4.4% at April 2020, then more than triple at peak of 14.7% in April 2020 and maintain a stable rate at
more than 10%. 

```python
Image(filename='Data/Image/UnemploymentRate-2009.png') 
```

Compare to the Great Recession which occurs from 2007 to 2009, the unemployment rate is smaller, with
only a mean from January 2009 to December 2009 at 9.283%. Since the population of the United States was around 306 millions [4]
compare to 331 millions by today, the number of non-employed people is significantly higher. 
Since the COVID-19 situation in the United States is not improving, [5] newly unemployment people
are not actively looking for work, thus making them not count as unemployed in
statistic. Due to the loss of job and work labors, it creates a negative effects
on economic. More unemployment-rate and WFH lead to a reduction of demand on consumption good due to 
lower on salary, and the supply chain cannot shrink down suddenly to adapt the situation, thus
making a surplus in production.


```python
Image(filename='Data/Image/NYSE-2020.png') 
```

It is fairly natural the stock market react much faster than the COVID-19 Financial Crisis.
According to the NYSE Composite chart provided by Yahoo!Finance, at March 23 2020,
the NYSE index reduced from 14,014 USD (2/21/2020) to approximately 9,000 USD, account for
a loss of 35% in just a month. It was a worst drop since the Great Recession. It is safe to say that
the stock market is a reflection of panic psychology, which is a natural respond from
the pandemic. At the time of the 2020 Stock Market crash, the unemployment-rate was still around 4.4%.
Mass hysteria was the main theme from mid-Febuary to April, with the double increase in volume.

Overall, the COVID-19 creates a tremendous effect in both supply chain, consumption demand, 
and panic selling effect in stock market by force the population to conduct social distancing.

```python
import numpy as np
a = np.array([1,2,3,4,5])
np.mean(a)
```

```python
%load -s Hello test.py
# haha
```

It is quite good that we can 

## References
[1] https://www.who.int/news-room/q-a-detail/q-a-how-is-covid-19-transmitted

[2] https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/social-distancing.html

[3] https://tradingeconomics.com/united-states/unemployment-rate

[4] https://www.worldometers.info/world-population/us-population/

[5] https://www.nber.org/papers/w27017.pdf
