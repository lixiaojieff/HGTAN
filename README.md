# HGTAN

Implementation of Paper：Temporal-Relational Hypergraph Tri-Attention Networks for Stock Trend Prediction

## Abstract
Predicting the future price trends of stocks is a challenging yet intriguing problem given its critical role to help investors make profitable decisions. In this paper, we present a collaborative temporal-relational modeling framework for end-to-end stock trend prediction. The temporal dynamics of stocks is firstly captured with an attention-based recurrent neural network. Then, different from existing studies relying on the pairwise correlations between stocks, we argue that stocks are naturally connected as a collective group, and introduce the hypergraph structures to jointly characterize the stock group-wise relationships of industry-belonging and fund-holding. A novel hypergraph tri-attention network (HGTAN) is proposed to augment the hypergraph convolutional networks with a hierarchical organization of intra-hyperedge, inter-hyperedge, and inter-hypergraph attention modules. In this manner, HGTAN adaptively determines the importance of nodes, hyperedges, and hypergraphs during the information propagation among stocks, so that the potential synergies between stock movements can be fully exploited. Extensive experiments on real-world data demonstrate the effectiveness of our approach. Also, the results of investment simulation show that our approach can achieve a more desirable risk-adjusted return.

## Overall Architecture

![Image text](https://github.com/lixiaojieff/HGTAN/blob/main/framework.png)

## Dataset

### Historical Price Data
We used a financial data API-https://tushare.pro to collect the historical price and relational data of stocks from China’s A-share market. We further performed a filtering step to eliminate those stocks that were traded on less than 98% of all trading days. This finally results in 758 stocks between 01/04/2013 and 12/31/2019.  

### Relation Data
We considered the industry-belonging and fund-holding relationships of stocks. For the former, we grouped all stocks into 104 industry categories according to the Shenwan Industry Classification Standard. For the latter, we selected 61 mutual funds established before 2013 in the A-share market, and acquired the constituent stocks of each fund from the quarterly portfolio reports.  

## Models

  * `/HGTAN/models.py`: end-to-end prediction framework;
  * `/HGTAN/temp_layers.py`: temporal attention layer;   
  * `/HGTAN/hyperedge_attn.py`: implementation of intra-hyperedge attention module;   
  * `/HGTAN/tri_attn.py`: inter-hyperedge attention module and inter-hypergraph attention module;   
  * `/HGTAN/layers.py`: implementation of attention layer;   


## Requirements

Python >= 3.6  
torch >= 1.4.0  
torchvision >= 0.1.8  
numpy  
sklearn  
  
## Hyperparameter Settings

Epoch: 600  
BatchSize: 64  
Learning Rate: 1e-3  
Dropout: 0.5
  
 ## Contact
 
If you have any questions, please contact lixiaojie199810@foxmail.com.
