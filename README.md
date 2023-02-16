# Spotify & Billboard Hot 100<br>
## Project Description<br>
For this project I gathered Spotify music data, analyzing it for particular audio features that will later be used as drivers for time on Billboard's chart. 

## Project Goals<br>
- Discover drivers of time on charts
- Use drivers to develop a machine learning model that accurately predicts chart time
- This information could be used on future datasets to help find future hits

## Initial Questions<br>
- Does loudness affect time on chart?
- Does more energy mean longer time on chart?

## The Plan<br>
#### Acquire data
* Data was acquired from Kaggle & Github
* Left merged Billboard dataset using unique key combined from features
* Spotify dataset contained 2000 samples and 20 features
* Billboard Hot 100 dataset contained 336_295 samples and 12 features
* Each row represents a song consindered a hit
* Each column represents a feature of those songs

#### Prepare
**Prepare Actions:**
* Removed columns that did not contain useful information
* Cleaned data due to complex naming
* Fixed Nulls From Key Errors
* Checked that column data types were appropriate
* Fixed incorrect datatypes
* Encoded categorical bi-variate
* Split data into train, validate and test


## Data Dictionary<br>
| Name             | Definition |
| :--------------- | :--------- |
| duration_ms      | Duration of the track in milliseconds |
| energy           | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.  |
| loudness         | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db. |
| valence          | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). |
| acousticness     | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. |
| year             | Release Year of the track |
| explicit         | The lyrics or content of a song or a music video contain one or more of the criteria which could be considered offensive or unsuitable for children. |
| instrumentalness | Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. |
| mode             | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.  |
| speechiness      | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. |

## Steps to Reproduce
- Clone this repo.
- Acquire the data from Kaggle & Github
- Put the data in the file containing the cloned repo.
- Run notebook.

## Takeaways and Conclusions

* explicit, energy, loudness were key drivers for time on chart

## Recommendations
* Explore different approach to loudness and energy instaead of compression

## Next Steps
* Look for combination of features to explore correlation
* Treat Target as discrete by binning


