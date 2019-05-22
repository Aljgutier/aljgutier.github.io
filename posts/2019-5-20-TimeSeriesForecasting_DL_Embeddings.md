---
title: "Time-series Deep Learning Embeddings"
date: "2019-05-3 18:08:41 +0000"
author: Al Gutierrez
description: Time series forecasting, Deep Learning, Embeddings, Random Forest,  Fastai, Pytorch, Jupyter Notebook
...
<h1 style="color:	#115BDC;">Time Series Forecasting with Deep Learning and Embeddings</h1>


<figure>
 <img alt="Time-series graph, Rosmman, top and bottom store sales" title="Time-series graph, Rosmman, top and bottom store sales" src="/images/TimeSeriesForecasting/TSeriesChart.png" width="635">
 <figcaption><center>Rossman Top (Store 262) and Bottom (Store 307) Store Sales Time-Series</center></figcaption>
 </figure>

In this article we study a state-of-the-art predictive analytics pipeline for structured data. Structured data is also known as "tabular data" and represents the most common data format in the industry. Though it is well-known that deep learning has achieved significant breakthroughs for unstructured data, suach as in computer vision and NLP, it is not as widely known that deep learning, with the use of ([Embeddings](https://www.fast.ai/2018/04/29/categorical-embeddings/)), can provide significant predictive performance improvement for structured data. Below, we walk through the Python code based on the [Fastai](https://www.fast.ai/) library demonstrating how to set up a predictive analytics pipeline based on deep learning with embeddings. We utilize the Kaggle, Rossmann dataset, discuss the deep-learning architecture, training, performance, and compare the performance to a machine learning tree-based model. 


#### Use Cases

Forecasting and prediction problems include a broad set of use cases, such as

* stock price/index forecasting
* sales forecasting
* demand forecasting
* price prediction
* customer churn prediction
* sentiment prediction
* risk scoring
* customer segmentation demand prediction



### Predictive Analytics and Forecasting

In this article, we solve a forecasting problem. However, the approach applies equally to both forecasting and prediction. Forecasting involves predicting the future value of the dependent variables based on historical independent variables and or the historical dependent variable (auto-regressive). The forecasting problem is a subset of the prediction problem, wherein the prediction problem does not include a time component. 

In the prediction or forecasting problem, the pre-processing stage consists of independent variable transformations including aggregations, time differences, and lags to produce a single row of data with potentially many independent variables corresponding to one or more dependent variables. For the forecasting problem, this single row is suitable for forecasting the dependent variable. Although there is a significant dependence on properly engineering the model features (independent variables), after feature engineering the predictive model algorithm structure is the same for prediction or forecasting. Thus, the methods herein apply to both types of problems


### Rossmann Dataset

The Rossman, Kaggle dataset is chosen for the following reasons. A dataset with benchmarked performance with realistic complexity is preferred. For example, there are several well-known data sets for the development of AI and ML models. Some examples include CIFAR for image processing, Wordnet for NLP, the Iris data set, and Handwritten digits dataset. These datasets tend to be excellent for exploring data algorithms because the data science community understands the data and there are numerous published examples.

Similarly, the Rossmann, Kaggle dataset is becoming popular for exploring forecasting problems; for example, it is referenced in the following use cases. 

 * [Rossmann store sales competition](https://www.kaggle.com/c/rossmann-store-sales)
 * [Fastai introduction to Deep Learning for Tabular Data](https://www.fast.ai/2018/04/29/categorical-embeddings/)
 * [Stanford CS229, Final Projects based on Rossman store sales](http://cs229.stanford.edu/projects2015.html)
 * [Journal publications, Entity Embeddings of Categorical Variables](https://arxiv.org/pdf/1604.06737.pdf)
 
Additionally, the Rossman, Kaggle dataset is the only open source time-series dataset with several published evaluations and benchmarks by the data science community. Utilizing this dataset enables focusing on the Deep Learning and Entity Embeddings while taking as a given pre-processing and basic understanding of the data that are established by other published exercises.

<h1 style="color:	#115BDC;">Data Processing and Machine Learning Overview</h1>

<figure>
 <img alt="Time-series forecast machine learning pipeline." title="Time-series forecast machine learning pipeline." src="/images/TimeSeriesForecasting/TSeriesForecastMLPipeline.png" width="635">
 <figcaption><center>Figure 1. Time-series forecasting machine-learning pipeline </center></figcaption>
 </figure>
 
Figure 1. illustrates the data processing pipeline including loading the raw data, X' (representing data prior to time T), pre-processing, preparing the data for machine learning and prediction, and forecasting of the future value ŷ for time t ≥ T. This article is primarily concerned with the 3rd step, the forecasting model and in particular entity embeddings within a deep-learning architecture. In order to appreciate the benefits of the deep-learning model, it is useful to compare it to an ensemble tree-based model, a Random Forest. 

This deep-learning architecture is adjustable to fit the problem (size and the number of the layers), but otherwise is structurally fixed. That's not to say that an alternate forecasting model is not effective. The first processing step, data pre-processing, is very much problem and domain specific. In the case of structured data, data pre-processing, including feature engineering, can have a significant impact on overall model performance. Like the DL forecasting architecture, the second step in the pipeline, "Preparing for Machine Learning," does not change with the domain and consists repeatable methods, such as normalization of continuous variables and numericalization of categorical variables.

The coding begins with importing the fastai library, which is based on Fastai version 0.7. See installations instructions here [Fastai 0.7](https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652). We also set a variable,`PATH` that points to the dataset. The data is available from Kaggle, and for convenience, can be downloaded in one .tgz file from [here](http://files.fast.ai/part2/lesson14/rossmann.tgz).

```python
from fastai.structured import *
from fastai.column_data import *
PATH='data/rossmann/'
```

<h1 style="color:	#115BDC;">Data Pre-Processing</h1>

Though we do not discuss the code for the data pre-processing transformations in detail, it is essential to understand the data, including the pre-processing and preparation for machine learning. These pre-processing methods serve as examples for other similar problems and domains. However, for this article, the feature engineering ("Pre-processing") is taken as a given.

Input to the pre-processing is from the tables listed below. Additional information regarding these tables are further discussed on the [Kaggle, Rossman, competion, data page](https://www.kaggle.com/c/rossmann-store-sales/data)


<table>
   <caption> Table 1. Input Data Tables </caption>
   <tr>
      <td style="text-align:center;vertical-align:center;"> Table </td>
      <td style="text-align:center;vertical-align:center;"> Columns </td>
   </tr>
   <tr>
      <td> train.csv </td>
      <td> Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday </td>
   </tr>
   <tr>
      <td> store.csv </td>
      <td> Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval </td>
   </tr>
   <tr>
      <td> store_states.csv </td>
      <td> Store, State </td>
   </tr>
   <tr>
      <td> state_names.csv </td>
      <td> StateName, State </td>
   </tr>
   <tr>
     <td> googletrends.csv  </td>
     <td> file,week,trend </td>
   </tr>
   <tr>
      <td> weather.csv </td>
      <td> file, Date , Max_TemperatureC , Mean_TemperatureC , Min_TemperatureC , Dew_PointC , MeanDew_PointC , Min_DewpointC , Max_Humidity , Mean_Humidity , Min_Humidity , Max_Sea_Level_PressurehPa , Mean_Sea_Level_PressurehPa , Min_Sea_Level_PressurehPa , Max_VisibilityKm , Mean_VisibilityKm , Min_VisibilitykM , Max_Wind_SpeedKm_h , Mean_Wind_SpeedKm_h , Max_Gust_SpeedKm_h , Precipitationmm , CloudCover , Events , WindDirDegrees </td>
   </tr>
   <tr>
      <td> test.csv </td>
      <td> Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday </td>
   </tr>
   
</table>

First, the data files are read into a Pandas dataframe. The use of unique data sources can potentially have a good payoff for predictive performance. In this case, Kaggle competitors employed [Google Trends to identify weeks and states correlated with Rossman](https://www.kaggle.com/c/rossmann-store-sales/discussion/17130) and weather informaiton. This information is transformed into machine learning features during the prprocessing step followed by normalization (continuous variables) and numericalization of categorical variables. The preprocessing is taken directly from the 3rd place competitor, and the code is contained in the fastai [lesson3-rossmann.ipynb]() Jupyter notebook. For convenience, a Jupyter notebook with only the preprocessing commands is available [here](https://github.com/Aljgutier/aljpspacedl1). The preprocessing employs typical methods for this type of data, for example, table joins, running averages, time until next event, and time since the last event. The outputs of the preprocessing section are saved in the "joined" and "joined_test" files in "feather" format, which are then loaded into the Jupyter notebook for the next step of processing ("Prepare for ML and Prediction").

```python
joined = pd.read_feather(f'{PATH}/joined')
joined_test = pd.read_feather(f'{PATH}joined_test')
```
The columns (i.e., variables) of the `joined` table include `Sales` the dependent variable and all other variables (independent variables). The independent variables within each row are processed from their original form in X' (as described in the previous paragraph) so that one row of independent variables is intended to predict the corresponding dependent variable, `Sales` (in the same row).

<h1 style="color:	#115BDC;">Prepare for Machine Learning</h1>

The next stage of the pipeline, "Prepare for ML and Prediction" takes as input the tabular data in the `joined` table. Machine learning algorithms perform better when the continuous value inputs are normalized and the categorical values are numericalized. Furthermore, artifcial neural networks require the inputs to be zero mean with standard deviation of 1.

### Categorical and Continuous Variables

The categorical variables and numerical variables are listed in the `cat_vars` list and `contin_vars` list.  These variables are selected from the `joined` table as features for machine learning and prediction.

```python
cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

contin_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']

n = len(joined); n
```

### Training, Validation and Test Set

Next, the training and validation sets are created, with use of the fastai `proc_df()` function, which performs the following functions.

* The dependent variable is put in into an array, `y` and independent variables are put into the dataframe, `df`.  
* Continuous variables are normalized (zero mean and standard deviation = 1), and categorical variables are enumerated.  
* Continuous variable missing values are filled with the median, and categorical id of 0 is reserved for categorical variable missing values. 
* The dictionary `nas` is a mapping of the N/A's and the associated median.
*  `mapper` is a DataFrameMapper, which stores the mean and standard deviation of the continuous variables and can be used for scaling during test time.

The `val_idx` variable are indexes identifying the part of `df` (training set) to be used for validation. In this forecasting problem, the validation indexes correspond to the same time frame (different year) as in the test set (from `joined_test`) and in this case the test set time frames are defined by Kaggle. 

```python
df, y, nas, mapper = proc_df(joined_samp, 'Sales', do_scale=True)
yl = np.log(y)

joined_test = joined_test.set_index("Date")

df_test, _, nas, mapper = proc_df(joined_test, 'Sales', do_scale=True, skip_flds=['Id'],
                                  mapper=mapper, na_dict=nas)
val_idx = np.flatnonzero(
    (df.index<=datetime.datetime(2014,9,17)) & (df.index>=datetime.datetime(2014,8,1)))                                  
                                  
```

### Optimization Function and Log y 

One additional transformation is necessary prior to machine learning. The optimization function for evaluation of results is RMSPE (Root Mean Square Percentage Error). The Kaggle submissions are evaluated on the Normalized Root Mean Squared Percent Error (RMSPE), calculated as follows:

$$  \enspace\enspace   EQN-1 \hspace{3em} RMSPE =   \sqrt{\frac{1}{n}\sum_{k=1}^n \left(   \frac{y_i - ŷ_i}{y_i} \right) } $$


This metric is suitable when predicting values across a large range of orders of magnitudes, such as Sales. It avoids penalizing large differences in prediction when both the predicted and the true number are large: for example, predicting 5 when the true value is 50 is penalized more than predicting 500 when the true value is 545.

The RMSPE metric is not directly available from machine learning libraries. For example, it is not availble form Fastai, PyTorch, or Sklearn. The optimization function typically available for regressor functions is RMSE (Root Mean Square Error). Therefore, it is typical to use log properties as follows.

For example, for $$ \sum_i{ \frac{ŷ_i } {y_i} } 
$$ we use the property  $$ln(\frac{a}{b}) = ln(a) - ln(b)$$, and therefore

$$ 
  \enspace\enspace   EQN-2 \hspace{3em} \sum_i{ ln \left(  \frac{ŷ_i} {y_i} \right)  } = \sum_i{\left(ln(ŷ_i) - ln(y_i)   \right)} 
$$           

The implication is that when we take the log of the dependent variable y, then we get RMSPE for free and taking the inverse log, that is $$exp^{log(RMSPE)}$$ we get the RMSE.

For convenience, we define the following functions.

```python
def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)
```


<h1 style="color:	#115BDC;">Tree based model (Random Forest)</h1>

At this point, the data is ready for machine learning. Training a Random Forest model is is useful to establish a performance baseline. The dependent variable is the log of Sales (yl) and the independent variables are in the df Dataframe. The indexes in `val_idx` are used to split between training and validation. A Random Forest regressor with 40 estimators, 2 samples per leaf yields an RMSPE score of 0.1086. It is worthwhile noting that the [tree models do not require one-hot encoding of categorical values](https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/), because they operate on the concept of partitioning the decision space. Since at this point the categorical variables are numerical values the model can operate directly on the `df` dataframe. 

```python
from sklearn.ensemble import RandomForestRegressor
```
```python
((val,trn), (y_val,y_trn)) = split_by_idx(val_idx, df.values, yl)
mrf = RandomForestRegressor(n_estimators=40, max_features=0.99, min_samples_leaf=2,
                          n_jobs=-1, oob_score=True)
mrf.fit(trn, y_trn);
preds = mrf.predict(val)
mrf.score(trn, y_trn), mrf.score(val, y_val), mrf.oob_score_, exp_rmspe(preds, y_val)
```


    (0.9821234961372078,
    0.9316993731280493,
    0.9241039979971587,
    0.10863755294456862)


<h1 style="color:	#115BDC;">Deep Learning with Embeddings</h1>

<figure>
 <img alt="Deep learning neural network time-series forecasting arhchitecture" title="Deep learning neural network time-series forecasting arhchitecture"  src="/images/TimeSeriesForecasting/DLNN_Forecast_wEmbeddings.png" width="635">
 <figcaption><center>Figure 2. Deep-learning neural network time-series forecasting architecture</center></figcaption>
 </figure>

The deep learning (DL) with Embeddings architecture is illustrated in Figure 2. The DL model receives as input the feature variables generated from the previous stage of processing and is comprised of continuous and categorical variables, `(cvs ... cats...)`. The architecture is based on the [Fastai embeddings for structured data](https://www.fast.ai/2018/04/29/categorical-embeddings/) architecture. The first layer is an embeddings layer followed by two  fully connected layers, and then the output layer. 

### Deep Learning Model

The deep learning model is defined with the Fastai ColumnarModelData() class. It receives as input the `PATH`,  validation indexes, Numpy array of independent variables (`df.values`),  batch size, and test set (`df_test`). Next, a `learner` object is created with `md.get_learner()`. The learner receives as input the size  ("cardinality") of each categorical variable.


```python
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128,
                                       test_df=df_test)

cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]

emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]

# Inputs to the learner:
#    No. of continuous variables =  total Variables - Categoricals. 
number: Sales
#    Categorical variable dropouts is set to .04
#    Output of the last linear layer, is 1, for predicting single output value
#    Activations in first and second linear layers ... [1000,500]
#    Dropout in first and second linear layers ... [.001,0.1]
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3
```

As illustrated in the diagram, there is an embeddings table for each categorical value. The embeddings are setup and work as follows:

* By rule of thumb, the width of each table is sized proportionally to the cardinality of the corresponding categorical variable (cardinality divided by 2 plus 1) up to a width of 50. The value 0 is reserved for unknown. 
* The number of rows is the cardinality of the categorical variable. 
* Each categorical variable serves as a lookup to the i-th row of values (ek\_ij) corresponding to the k-th categorical variable. 
* The output from each table, embeddings row, are effectively concatenated with the continuous values to form the input into the fully connected layers. 
* The embedding values (ek_ij) are optimized as part of the deep-learning model stochastic gradient descent training. 
  
Below are listed the width's, `cat_sz`, for each embeddings table.

The first layer of the fully connected layer is set to 1000 activations with ReLU non-linear function, and the second fully connected layer is set to 500 activations (also ReLU), and 1 sigmoid output. The output range is defined with y-range.


```python
cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars]
cat_sz
```


     [('Store', 1116),
     ('DayOfWeek', 8),
     ('Year', 4),
     ('Month', 13),
     ('Day', 32),
     ('StateHoliday', 3),
     ('CompetitionMonthsOpen', 26),
     ('Promo2Weeks', 27),
     ('StoreType', 5),
     ('Assortment', 4),
     ('PromoInterval', 4),
     ('CompetitionOpenSinceYear', 24),
     ('Promo2SinceYear', 9),
     ('State', 13),
     ('Week', 53),
     ('Events', 22),
     ('Promo_fw', 7),
     ('Promo_bw', 7),
     ('StateHoliday_fw', 4),
     ('StateHoliday_bw', 4),
     ('SchoolHoliday_fw', 9),
     ('SchoolHoliday_bw', 9)]


### What's the big idea with Embeddings? 

The performance gains of deep Learning, concerning forecasting, is primarily attributed to entity embeddings. Entity embeddings are a low-dimensional representation of the high dimensional space of categorical variables. Consider, for example, the following intuition. A common technique for representing a categorical variable is one-hot encoding.  For the case of a movie title one-hot encoding quickly results in a sparse representation consisting of a row with thousands of columns in width with one non-zero element representing one movie title.

In contrast, an entity embedding represents the relationship between movies with a low-dimension vector, such as genre. In this case, a movie may be represented with a vector representing genres:  western, action, drama, sci-fi, cinematography, etc. When input to a machine learning algorithm, these low dimension entity embeddings representations enable the algorithm to exploit relationships between categorical elements. 

### Fit

With the model defined, it is time to fit. The learning rate is set to 1e-3. Though not demonstrated here, it is useful to mention that the Fastai library includes the `lr_find()` method to find an [optimal learning rate](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-1-602f73869197) based on the paper [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf).

```python
m.fit(lr, 1, metrics=[exp_rmspe])
```

    epoch      trn_loss   val_loss   exp_rmspe                       
        0      0.013699   0.013854   0.116042  

The model is trained with several epochs at a constant learning rate.

```python
m.fit(lr, 3, metrics=[exp_rmspe])
```

    epoch      trn_loss   val_loss   exp_rmspe                       
        0      0.011076   0.015949   0.113352  
        1      0.009257   0.012345   0.10875                          
        2      0.008702   0.011414   0.102238     
                            
Next, the cycle_len =1 parameter enables [SGDR with restarts](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-2-eeae2edd2be4), whereby the learning rate is decreased with Cosine Anealing profile over a `cycle_len`measured in Epochs. Following a cycle the learning rate is returned to `lr` to begin another cycle.

```python
m.fit(lr, 5, metrics=[exp_rmspe], cycle_len=1)
```

    epoch      trn_loss   val_loss   exp_rmspe                        
        0      0.007108   0.010651   0.098073  
        1      0.00664    0.010301   0.096721                         
        2      0.007111   0.010438   0.096768                         
        3      0.006871   0.010481   0.096761                         
        4      0.006245   0.010313   0.096277     

After 5 epochs we obtain an RMSPE of 0.0963 representing approximately 11% improvement over the Random Forest. The Random Forest, though receives as input the numericalized categories, does not employ embeddings.


### Test

On the test data we obtain an RMSPE of 0.0998. 

```python
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128,
                                       test_df=df_test)
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
m.load('val0')
x,y=m.predict_with_targs()
print(exp_rmspe(x,y))
pred_test=m.predict(True)
pred_test = np.exp(pred_test)
pred_test
```

    0.09979241514057972


### Predict

Below, is illustrated how to load a saved model and apply it to a new set of data. One additional step necessary for new data is to pre-process and prepare it for prediction, with the same pre-processing and preparation functions as the training and test sets. For convenience, the test data is used below (already pre-processed).

```python
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128,
                                       test_df=df_test)
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
m.load('val0')

# columnar dataset
cds = ColumnarDataset.from_data_frame(df_test,cat_vars)
# data loader
dl = DataLoader(cds,batch_size=128)
# log predictions
predictions = m.predict_dl(dl)
# exp (log predictions) = predictions
predictions=np.exp(predictions)
predictions
```



       array([[ 4475.422 ],
       [ 7242.9165],
       [ 9095.177 ],
       [ 7343.125 ],
       [ 7703.5986],
       [ 5933.637 ],
       [ 7425.5747],
       [ 8448.557 ],
       ...,

<h1 style="color:	#115BDC;">Summary and Conclusions</h1>

In summary, the deep learning with embeddings model produces world-class predictive performance on the Rossman dataset. The model achieves a significant improvement as compared to a Random Forest (RF) model, the next best model as reported in [Entity Embeddings of Categorical Variables](https://arxiv.org/pdf/1604.06737.pdf). Though it is significantly more complex than the RF model, the training time is approximately the same on a basic GPU vs. Random Forest on CPU. The key differentiating method for achieving the performance gain is the use of entity embeddings. Entity embeddings are a low-dimensional representation of the high dimensional space of categorical variables and enable machine learning algorithms to exploit relationships between categorical items.

<table>
 <caption> Table 1. Summary of Deep Learning time-series forecasting model</caption>
 <tr>
   <td style="text-align:center;vertical-align:top;"><Strong>Characteristic </Strong> </td>
   <td style="text-align:center;vertical-align:top;"><Strong>Description</Strong></td>
 
 </tr>
 <tr>
   <td>Machine Learning Pipeline </td>
   <td>- Data Pre-processing <br>
       - Prepare data for ML and Prediction <br>
       - Machine Learning and Prediction <br>
 </tr>
 <tr>
   <td>Architecture</td>
   <td>Deep learning with embeddings <br>- Embeddings layer <br> 
      - 2 Fully connected layers, 1000 and 500 ReLU activations <br>
      - Output layer 1 sigmoid activation <br>
   </td>
 </tr>
 <tr>
   <td> Data Set </td>
   <td> - Rossman, Kaggle dataset <br>
        - Pre-processing steps taken from 3rd place winner <br>
        - 844,438 raining rows, 22 categorical features, 16 continuous features
     </td>
 </tr>
   <td> Training </td>
   <td> - Paperspace P4000 virtual desktop: NVIDIA P4000, 8 GB GPU, 1791 CUDA Cores, and 30 GB, Intel Xeon E5-2623 v4 CPU <br>
        - ~10 minutes  <br>
        - SGDR with restarts and cosine anealing
    </td>
</table>



