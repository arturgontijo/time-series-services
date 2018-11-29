[issue-template]: ../../../issues/new?template=BUG_REPORT.md
[feature-template]: ../../../issues/new?template=FEATURE_REQUEST.md

![singnetlogo](../../assets/singnet-logo.jpg?raw=true 'SingularityNET')

# CNTK Finance Next Day Trend

This service uses [CNTK Finance Timeseries](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_104_Finance_Timeseries_Basic_with_Pandas_Numpy.ipynb) 
to perform a time series analysis.

It is part of our third party [Time Series Analysis Services](https://github.com/singnet/time-series-analysis).

### Welcome

The service receives an image and uses it as an input for a pre-trained `ResNet152` model.

### What’s the point?

The service makes prediction using computer vision and machine learning techniques.

The service outputs a top 5 prediction list (ordered by confidence) based on the specified dataset (flowers or dogs).

### How does it work?

The user must provide the following inputs in order to start the service and get a response:

Inputs:
  - `source`: Source to get market data (ie. yahoo, check this [link](https://github.com/pydata/pandas-datareader/blob/master/pandas_datareader/data.py#L306)).
  - `contract`: Label of asset (like "SPY").
  - `start_date`: Start date of training dataset.
  - `end_date`: End date of training dataset.
  - `target_date`: Date that will be analysed.
  - The date delta must be >= 100 days.

You can use this service from [SingularityNET DApp](http://alpha.singularitynet.io/), clicking on `SNET/ImageRecon`.

You can also call the service from SingularityNET CLI (`snet`).

Assuming that you have an open channel (`id: 0`) to this service:

```
$ snet client call 0 0.00000001 54.203.198.53:7009 trend '{"source": "yahoo", "contract": "SPY", "start_date": "2017-01-01", "end_date": "2017-10-31", "target_date": "2018-11-28"}'
...
Read call params from cmdline...

Calling service...

    response:
        
```

### What to expect from this service?

Input:



Response:
```

```

Input :


Response:
```

```