# S-MA
This is the repository of A Transformer-based Short-Term Memory Attention model for Multimodal Sentiment Analysis
![image](https://github.com/Doyken/S-MA/blob/main/S-MA.svg)
# Requirements
We give the version of the python package we used, please refer to `requirements.txt`
The code will released soon.  
  `python == 3.9`  
  `pytorch == 1.8.1`  
  `transformer == 4.10.0`  
  `scikit-learn == 0.24.2`
# Dataset
We provide text data in `dataset\data\`. As for images, please download from the link below:  
MVSA-* `http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/ ` 


HFM `https://github.com/headacheboy/data-of-multimodal-sarcasm-detection`

For example, MVSA-Single's images should be put in `dataset\data\MVSA-single\dataset_image`
# Run Code
Download model:
百度网盘：


`链接：https://pan.baidu.com/s/1v6gUz3dDj3uhglJnmQAs1Q` 
`提取码：cxz1`


Google drive:


`https://drive.google.com/drive/folders/1CLHGGq-NTeEUZsNhy651MTmBmKFyrcNG?usp=sharing`


If you want to run the code, just select the `'main.py'` file.
# Short-Memory Attention 
Here is the core of Short-Memory Attention. For more details, please refer to the 'model.py'.

    def forward(self,x,index ,now_epoch): 

       x= self.multiAttn(x, x, x) 
       
       if (self.mode == 0):  
       
                if now_epoch ==0:  
                
                    HT_t = x  
                   
                    self.CO_collection_tmp.append(x)  
                   
                else:  
                    if (index == 0):  
                        self.CO_collection = self.CO_collection_tmp  
                        self.CO_collection_tmp = []  
                    self.CO_collection_tmp.append(x)  

                    co_tminus1 = ( self.CO_collection_tmp[index] + x) / 2  
                    HT_t =(F.tanh(torch.bmm(x, F.tanh((co_tminus1.transpose(1, 2))))))  
                    dim_size = HT_t.size(2)  
                    linear_layer_HT_t= linear(dim_size)  
                    linear_layer_HT_t =linear_layer_HT_t.to(self.device)  
                    HT_t=linear_layer_HT_t(HT_t)  

                return HT_t  
        else:  
            return x  
