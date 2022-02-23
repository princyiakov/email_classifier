# Email Classifier
Remember how we have to click and classify our work mails based on themes ? 🤔 Dont we all get lost at what theme should 
the mail be classified into ? And worst when we have to retrive the important mails and we forgot to put them in a folder incase we missed to assign a theme !  🤕 

What if I tell you that we can automate this ? 

All you need is some data and some NLP ! 	🎊

This is exactly what we are trying to achieve via this project . Making your life simple one project at a time 	😄

<!-- toc -->

- [More About The Project](#more-about-the-project)
  - [Notebook File](#notebook-file)
  - [Understanding Data](#understanding-data)
  - [How did we solve the issue](#how-did-we-solve-the-issue)
     - [Data Cleaning](#data-cleaning)
     - [Data Visualization](#data-visualization)
     - [Feature Engineering](#feature-engineering)
     - [Model Selection](#model-selection)
    
<!-- tocstop -->

## More About The Project
Let us understand in detail about the data

### Notebook File 

Here is a quick view of my journey to achieve the goal :
https://github.com/princyiakov/email_classifier/blob/main/Email%20Classifier.ipynb

### Understanding Data 

Classification labels:

    1. `regulations and regulators (includes price caps)`
    2. `internal projects -- progress and strategy`
    3. `company image -- current`
    4. `company image -- changing / influencing`
    5. `political influence / contributions / contacts`
    6. `california energy crisis / california politics`
    7. `internal company policy`
    8. `internal company operations`
    9. `alliances / partnerships`
    10. `legal advice`
    11. `talking points`
    12. `meeting minutes`
    13. `trip reports`

Mail Sample : 
```2;internal company operations;"Message-ID: <8687721.1075852656109.JavaMail.evans@thyme>
Date: Wed, 13 Jun 2001 06:05:07 -0700 (PDT)
From: steven.kean@enron.com
To: kenneth.lay@enron.com, jeff.skilling@enron.com
Subject: Enron Expatriates in India
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Kean, Steven </O=ENRON/OU=NA/CN=RECIPIENTS/CN=NOTESADDR/CN=B86DC838-ECAFDC13-8625693C-57FA6D>
X-To: Lay, Kenneth </O=ENRON/OU=NA/CN=RECIPIENTS/CN=KLAY>, Skilling, Jeff </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JSKILLIN>
X-cc: 
X-bcc: 
X-Folder: \JSKILLIN (Non-Privileged)\Deleted Items
X-Origin: Skilling-J
X-FileName: JSKILLIN (Non-Privileged).pst

John Brindle, David Cromley and others in the Corporate Business Controls group (formerly part of EBS) have been working over the last several weeks to make sure that we can get our people out of Dabhol if there are threats to their safety.  While no plan is perfectly reliable (particularly during monsoon season) we believe that we are as ready as we can be.
---------------------- Forwarded by Steven J Kean/NA/Enron on 06/13/2001 08:00 AM ---------------------------
From:	David Cromley@ENRON COMMUNICATIONS on 06/12/2001 09:18 AM
To:	Steven J Kean/NA/Enron@Enron
cc:	 

Subject:	Enron Expatriates in India

Steve,

I wanted to give you the latest figures on Enron expatriates in India.  There are currently 19 Enron expatriate employees based in India, accompanied by six dependants.  In addition, Enron has responsibility for 15 expatriate contractors working for DPC and Lingtec, primarily at Dabhol.  The total of 40 expatriates for whom Enron is directly responsible is broken down by location as follows:  21 in Mumbai, 16 at Dabhol, and three in Baroda.  By company, they are broken down as follows:  Enron India -  eight plus three dependants; EOGIL - nine plus one dependant; EBS - one plus two dependants; DPC - seven; and Lingtec - nine.  These figures are likely to continue to trend downward; Jane Wilson's departure on Friday, for example, will reduce the Enron India expatriate presence to seven plus three dependants.

We are confident that the evacuation plan now in place provides the best possible prospect of successfully evacuating all of these expatriates in the event of danger.  As a contingency, the plan also includes the evacuation of all 137 expatriates working at the Dabhol site, although a large-scale evacuation from Dabhol is more problematic, especially now that the monsoon season has started.  A coordinated plan was needed since the limited number of available helicopters meant that the evacuation plans drawn up by individual contractors such as Bechtel were all drawing upon the same resources.

As you know, Enron's security situation in India remains stable.  We are in the final stages of implementing security upgrades for our employees and offices there.

Dave
```

### How did we solve the issue 

#### Data Cleaning 

Performed different data cleaning techniques for NLP. 
- Basic cleaning of data from unique characters
- Added new stop words which improved the model performace
- Removed URLs 
- Lowercase the data
- Explored stemming and lemmatization where model performed better on using lematization

#### Data Visualization

Every Data has a story to tell . Let's understand the story behind this one !

- TARGET THEME DISTRIBUTION : As we see, the data is not equally distributed . We have ample amount of data to train for "reguations and regulators" 
  but only four data for "trip reports" ! 
  ![Target data disribution](https://github.com/princyiakov/email_classifier/blob/main/images/target_distribution.PNG)

- MAILS DISTRIBUTION BASED ON MONTHS : Maximum mails have been sent in the month of July and least in January
  ![Monthly data disribution](https://github.com/princyiakov/email_classifier/blob/main/images/month_distribution.PNG)

- What were they discussing during these two months ?! Lets see !
  July was more about "regulations and regulators (includes price caps)" and "california energy crisis / california politics"
  
  January was relaxed and more about understanding the future plans as the mails were "internal projects -- progress and strategy" and "legal advice" 
  
  ![Theme Month data disribution](https://github.com/princyiakov/email_classifier/blob/main/images/theme_for_month_distribution.PNG)
