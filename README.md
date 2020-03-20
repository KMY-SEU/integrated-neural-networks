# Basic Information
author  : kangmingyu1995<br>
e-mail  : kangmingyu.china@gmail.com

----
# Introduction
    It's essential to declare at first, it's an integrated neural netwok framework based on keras, whose backend is tensorflow. And, all these codes are originally collected from other authors in the github, such as tslgithub, titu1994 and flyyufelix. However, i have modified these codes about 40%-50%. Just in case, if you are one of the creators of these source code and you mind it, you can contact me, and i will delete them.  

    Secondly, I have made much improvements to previous codes, include
    - Use centralied train function to implement data reading and model training.
    - Use config.py to manage config parameters unifiedly. 
    - Update the version of keras, such as updating some methods and their parameters. If not update, this codes will not run sucessfully on new version of keras.
    - Replace ZeroPadding and Scale operations with Convolution in padding mode 'same'. I think it will make codes leaner.
    - By the way, i have corrected some basic structural errors in original model, which exploded my memory as first.
    - I have removed the support for theano from original model because these is experimental codes. I just use it to do control experiments of classification tasks, hence, no consideration is given to compatibility. Very sorry.    

    Anyway, for conclusion, the main idea for uploading these codes is to provide ready-made codes for researchers who need to do control experiments. It will help you to achieve results you want quickly. Meanwhile, if you are a beginner of deep learning, you also can start your learning journey from these simplified codes. All network structures are the same as the original paper.

    Welcome to discuss.
---

# Configuration
python == 3.7 
keras == 2.3.1
tensorflow == 1.14.0
opencv-python == 4.1.0.25
numpy == 1.17.1







