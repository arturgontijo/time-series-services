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
  - `Source`: Source to get market data.
  - `Contract`: Label of asset (like "SPY").
  - `Start Date`: Start date of training dataset ("2015-09-31").
  - `End Date`: End date of training dataset ("2018-09-31").
  - `Target Date`: Date that will be analysed ("2018-10-01").

You can use this service from [SingularityNET DApp](http://alpha.singularitynet.io/), clicking on `SNET/ImageRecon`.

You can also call the service from SingularityNET CLI (`snet`).

Assuming that you have an open channel (`id: 0`) to this service:

```
$ snet client call 0 0.00000001 54.203.198.53:7004 flowers '{"model": "ResNet152", "img_path": "https://www.fiftyflowers.com/site_files/FiftyFlowers/Image/Product/Mini-Black-Eye-bloom-350_c7d02e72.jpg"}'
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