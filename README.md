## Experiments:
Description | Validation split | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Public 
---| --- |---| --- |---| --- |---| --- | ---
cls embedding, 5 folds, 256 len|random|0.95177|0.95183 |0.96539|0.94426|0.94216|0.937931
cls embedding, 5 folds, 384 len|random|0.94733|0.95084  |0.96454 |0.94363|    |  
cls embedding, 5 folds, 256 len|5-folds|    |    |    |    |    |0.936347
4 hiddens embedding, 5 folds, 256 len|5-folds|0.94599|   |    |    |    |0.932137
cls embedding, 5 folds, 256 len, auxiliary variables, no reduce_lr |5-folds|0.95144|0.95352|0.94090|    |0.95581|  
cls embedding, 4 hiddens embedding, auxiliary variables, 5 folds, 256 len |5-folds|0.93830|0.93446|    |    |    |  
 pooled output embedding, auxiliary variables, 5 folds, 256 len|5-folds|0.93877|0.95207|0.95507|0.92468|0.95513|0.926533
  |    |0.93640|    |0.94749|0.94390|0.95887|0.929991|  
  seed = 1|    | 0.94941| 0.95266 |0.96144|0.94472|0.95867|    |  