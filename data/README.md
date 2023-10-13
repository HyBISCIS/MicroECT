# H5file Documentation 

Each file has n number of ECT readings. For example, the following readings are sorted by the timestamp  

``` 
['ECT_20220416_233341', 'ECT_20220416_233348', 'ECT_20220416_233355', 'ECT_20220416_233402', 'ECT_20220416_233408', 'ECT_20220416_233415', 'ECT_20220416_233422', 'ECT_20220416_233429', 'ECT_20220416_233435', 'ECT_20220416_233442', 'ECT_20220416_233458', 'ECT_20220416_233505', 'ECT_20220416_233512', 'ECT_20220416_233518', 'ECT_20220416_233525']

```

Each reading corresponds to a different column/row offset. Each reading has the following attrtibutes: 

```
<KeysViewHDF5 
['ADCbits', 'ADCfs', 'C_int', 'Nsamples_integration', 'PINOUT_CONFIG', 'T_int', 'V_CM', 'V_Electrode_Bias', 'V_STBY', 'V_STIMU_N', 'V_STIMU_P', 'V_SW', 'Vdd', 'col_offset', 'comments', 'dac_vfs', 'f_master', 'f_sw', 'row_offset', 'samplerate', 'timestamp', 'vectorfrequency']>

```

Each file has two image data: 

```
image_2d_ph1
image_2d_ph2
```


## Datset

-----------------------------------------------------------------------------------------------------------
Name      |     Slice Col   |      xmin, xmax       |      ymin, ymax      | bead_location | bead_diameter
------------------------------------------------------------------------------------------------------------
04172022  |       135       |         [24,39]       |         [120,150]    |     -20e-6    | 20e-6
------------------------------------------------------------------------------------------------------------
