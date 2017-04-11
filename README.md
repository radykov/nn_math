# Greater than 
Binary greater than (1 if first value is higher else 0) trains quicker than a categorical vector ([1,0] if first value higher else [0,1]) for deeper networks.

Try this out yourself by reducing the DATA_SIZE variable in the greater_than_1_hot.py file and see the performance degrade.

If you make the network shallow it learns quite quickly on only a few samples too. This is probably because the function is so simple to map