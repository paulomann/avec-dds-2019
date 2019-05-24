## Facial Action Coding System

![Examples of Action Units](https://raw.githubusercontent.com/wiki/TadasBaltrusaitis/OpenFace/images/AUs.jpg)

Facial Action Coding System (FACS) is a system to taxonomize human facial movements by their appearance on the face. Movements of individual facial muscles are encoded by FACS from slight different instant changes in facial appearance. Using FACS it is possible to code nearly any anatomically possible facial expression, deconstructing it into the specific Action Units (AU) that produced the expression. It is a common standard to objectively describe facial expressions. 

OpenFace is able to recognize a subset of AUs, specifically: 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, and 45.

You can find more details about FACS and AUs [here](https://en.wikipedia.org/wiki/Facial_Action_Coding_System) and [here](https://www.cs.cmu.edu/~face/facs.htm)

## Intensity and presence of AUs

AUs can be described in two ways
- Presence - if AU is visible in the face (for example AU01_c)
- Intensity - how intense is the AU (minimal to maximal) on a 5 point scale

OpenFace provides both of these scores. For presence of AU 1 the column `AU01_c` in the output file would encode 0 as not present and 1 as present. For intensity of AU 1 the column `AU01_r` in the output file would range from 0 (not present), 1 (present at minimum intensity), 5 (present at maximum intensity), with continuous values in between.

*NOTE* that the intensity and presence predictors have been trained separately and on slightly different datasets, this means that the predictions of both might not always be consistent (e.g. the presence model could be predicting AU as not being present, but the intensity model could be predicting its value above 1).

## Extraction from images and extraction from videos

OpenFace is able to extract Action Units both from images, image sequences and videos. It can also do it in videos containing multiple people, however, this will not be as accurate as running it on a video of a single person. This is because OpenFace is not able to re-identify people in videos and person specific calibration is not performed.

### Individual Images

Use `FaceLandmarkImg` project and executable for this, it will output AU predictions for each image. Please note that the accuracy of AU prediction on individual images is not as high as that of AU prediction on videos because videos allow for person specific calibration. For more details of how to perform the extraction see [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments)

### Image sequences and videos containing one person

If you want to extract facial action units from image sequence or a video you should use `FeatureExtraction` project and executable for this. This will provide AU presence and intensity predictions for each frame in a video.

### Multiple person videos

If you want to extract facial action units from videos that contain multiple faces you should use `FaceLandmarkVidMulti` project. NOTE that the extracted AUs will not be as reliable as in the single person in the video case, due to person specific feature calibration and post-processing which is currently not supported in multi-face case.

### Static vs dynamic

OpenFace uses two types of AU prediction models - `static` and `dynamic`. The `static` models only rely on a single image to make an estimate of AU presence or intensity, while `dynamic` ones calibrate to a person by performing person normalization in the video, they also attempt to correct for over and under prediction of AUs. By default OpenFace uses static models on images and dynamic models on image sequences and videos.

However, some video sequences do not have much dynamic range - the same expression is held throughout the video, this means that post calibration will not be helpful and might in fact be harmful, for those video sequences I recommend using `-au_static` flag which tells OpenFace not to perform dynamic calibration and to use only static models for AU prediction.

## Training AU models

The code for training AU prediction models is available [here](https://github.com/TadasBaltrusaitis/OpenFace/tree/master/model_training/AU_training), it also includes a readme explaining how to go around with training the models.

## Datasets used for training AUs in OpenFace

The datasets that were used for training AU recognition models are as follows:
- [Bosphorus](http://bosphorus.ee.boun.edu.tr/)
- [BP4D from FERA2015](http://sspnet.eu/fera2015/)
- [DISFA](http://www.engr.du.edu/mmahoor/DISFA.htm)
- [FERA2011](http://sspnet.eu/fera2011/fera2011data/)
- [SEMAINE from FERA2015](http://sspnet.eu/fera2015/)
- [UNBC](http://www.pitt.edu/~emotion/um-spread.htm)
- [CK+](http://www.pitt.edu/~emotion/ck-spread.htm)
