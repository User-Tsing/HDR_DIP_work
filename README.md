Checked by STAssn                
HDR高动态范围图像重建与处理                
实现功能：HDR图像合成，图像增强（高斯/中值，均衡(RGB)），图像频域展示             
要合成的图像全放一个文件夹里，运行时在里面选中那个文件夹即可。图像不能合成是因为图像大小不匹配。            
问题：图像偏移合成会产生重影等现象，没有做图像配准；高斯噪声加的太强了增强效果不明显；图像增强采用了RGB分通道而不是HSI分通道（失误）。            
使用Python而非MATLAB是因为课程大作业要求使用Python语言进行设计，关键组件：PyQt（界面设计），OpenCV（图像处理）。            
上传是为了纪念曾经制作过这样一个课程项目。       

PS:         
List of participants:           
General Integrator: STAssn;       
UI&Process: Checked and designed by STAssn;      
HDR: Modified by: S Fu;          
Photos: Searched by C Ch'ang;         
Other partners: H Ku(C), M Ch'ang       
（为避免不必要的麻烦，以上均为化名）
