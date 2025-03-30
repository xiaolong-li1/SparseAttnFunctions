# SparseAttnFunctions  
**Efficient Implementations of Sparse Attention Mechanisms**  

This repository contains multiple optimized implementations of sparse attention techniques, specifically tailored for **CogVideo2B**. These modifications enhance computational efficiency and memory usage, making them ideal for large-scale video generation tasks.

### Key Features:  
• **Sparse Attention Variants**: Includes several approaches to sparse attention, such as windowed attention, block-sparse attention, and more.  
• **CogVideo2B Integration**: Customized to seamlessly integrate with the CogVideo2B framework, ensuring optimal performance.  
• **Efficiency**: Designed to reduce memory footprint and accelerate computation, especially for high-resolution video generation. The proposed **Combined Attention** method achieves the highest sparsity while maintaining minimal quality degradation.
### Use Cases:  
• Large-scale video generation tasks.  
• Applications requiring efficient attention mechanisms for long sequences.  

### TO DO 
• Try to do some compression on head_dim 



## 🎥 视频效果对比

<div align="center">

### 🔄 Merge Token 策略效果可视化<small style="color: #718096;">(于Cogvideo-2b测试)</small>
<span style="display: inline-flex; align-items: baseline; font-size: 0.98em;">
  <a href="https://arxiv.org/abs/2210.09461" 
     style="color: #2f855a; text-decoration: none; border-bottom: 1px dotted #68d391; padding-bottom: 1px;"
     title="Token Merging: Your ViT But Faster"
     target="_blank">
    (Token Merging: Your ViT But Faster<sup style="font-size: 0.75em; color: #718096; margin-left: 2px;">[1]</sup>)
  </a>
  <span style="color: #718096; margin-left: 12px;"></span>
</span>






<table style="width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 15px;">
  <!-- 第一行 -->
  <tr>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video src="https://github.com/user-attachments/assets/6d60c797-f556-4b31-9b26-74ad3762a4a7" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 1000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
       <video src="https://github.com/user-attachments/assets/807743be-4163-4dfa-a6b5-ac2a326db553" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 2000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video src="https://github.com/user-attachments/assets/43a0ef78-32ff-4aad-94e7-bf05e453cb7f" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 3000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
  </tr>

  <!-- 第二行 -->
  <tr>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video src="https://github.com/user-attachments/assets/decba662-3fa5-4303-b3e8-d48bb8616946" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 4000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
    <video src="https://github.com/user-attachments/assets/fe9ed822-919c-44b5-bcf0-9fa6359f49aa" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 5000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video src="https://github.com/user-attachments/assets/12bff894-0cc2-4195-82ab-5029ba08b589" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 6000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
  </tr>












  <!-- 第三行 -->
  <tr>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
       <video src="https://github.com/user-attachments/assets/02a42f2a-e201-4fb6-9f9e-d3bf989d3374" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 7000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video src="https://github.com/user-attachments/assets/2cb81c40-b333-4856-85f9-f57d815b22a6" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 8000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #ebf8ff; border: 2px solid #63b3ed; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
       <video src="https://github.com/user-attachments/assets/1f150e0b-90e0-4bf8-ac5d-4596f2266550" width="100%" controls autoplay loop></video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2b6cb0; font-weight: 700;">⭐ 原始基准</span><br>
        <small style="color: #90cdf4;">未合并token</small>
      </div>
    </td>
  </tr>
</table>

## 🎯 多策略效果对比 <small style="color: #718096;">(CogvideoX1.5-5b测试平台)</small>

<div align="center" style="margin: 2rem 0;">

<table style="width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 15px;">
  <!-- 表头 -->
  <tr>
    <th style="padding: 12px; background: #f0fff4; border-radius: 8px; text-align: center; width: 33%;">
      <div style="font-size: 1.1em; color: #2f855a;">
  🔥 Smart Attention
      <span style="font-size: 0.9em; margin-left: 6px;">
        <a href="https://arxiv.org/abs/2502.01776" 
          style="color: #38a169; 
                  text-decoration: none;
                  border-bottom: 1px solid #c6f6d5;
                  padding-bottom: 1px;
                  transition: all 0.2s ease;"
          title="查看 Sparse VideoGen 论文"
          target="_blank">
          (Sparse VideoGen)
          <sup style="font-size: 0.8em; 
                    color: #718096;
                    vertical-align: super;
                    margin-left: 2px;">[1]</sup>
        </a>
      </span>
        <div style="font-size: 0.9em; color: #718096; margin-top: 4px;">
          ▲ 理论计算量下降64%
        </div>
      </div>
    </th>
    <th style="padding: 12px; background: #ebf8ff; border-radius: 8px; text-align: center; width: 34%;">
      <div style="font-size: 1.1em; color: #2b6cb0;">
        ⚖️ 基准对照组
        <div style="font-size: 0.9em; color: #718096; margin-top: 4px;">原始计算量100%</div>
      </div>
    </th>
    <th style="padding: 12px; background: #f0fff4; border-radius: 8px; text-align: center; width: 33%;">
      <div style="font-size: 1.1em; color: #2f855a;">
        🔄 Combined Attention (new method)
        <div style="font-size: 0.9em; color: #718096; margin-top: 4px;">理论计算量下降70-80%</div>
      </div>
    </th>
  </tr>
  
  <!-- 视频对比行 -->
  <tr valign="top">
    <!-- Smart Attention -->
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/f09afe74-40b3-4362-8340-a8735a69b4ed" width="100%" controls autoplay loop></video>
    </td>
     <!-- 基准组 -->
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video src="https://github.com/user-attachments/assets/3d356b84-8d02-4167-b149-e430bbea31bf" width="100%" controls autoplay loop></video>
    </td>
    <!-- Combined Attention -->
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/3d80f62f-e85c-4d30-a9d7-ac77fe5346b3" width="100%" controls autoplay loop></video>
    </td>
   
  </tr>

  <!-- 重复其他3组视频对比 -->
  <tr valign="top">
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/09e7967a-bc1e-42d0-a2cd-cf4e537f2d70" width="100%" controls autoplay loop></video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/b50d7b7d-39fc-4e62-9af3-e817df2db022" width="100%" controls autoplay loop></video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/49eb8ff6-223a-47ad-90b8-e1bae7cc123c" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  
  <tr valign="top">
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/d6bdff00-6745-4103-8fac-08ee09309237" width="100%" controls autoplay loop></video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/62d2d654-6852-4f64-b50c-d8ea55b5545a" width="100%" controls autoplay loop></video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video src="https://github.com/user-attachments/assets/6abf658f-8ab9-437f-955b-2d6964582f8e" width="100%" controls autoplay loop></video>
    </td>
  </tr>
  
  <tr valign="top">
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/3aa89b79-6777-4d4f-8fdd-639cfdf64097" width="100%" controls autoplay loop></video>
    </td>
     <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/dd15be39-b4d9-4cae-8b1b-9d201bdd1002" width="100%" controls autoplay loop></video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
       <video src="https://github.com/user-attachments/assets/cea775b3-0364-4fe2-aa99-4936f8d56ac9" width="100%" controls autoplay loop></video>
    </td>
  </tr>
</table>

To view the corresponding prompt words for the gallery, please click [here](prompts.txt)
</div>
