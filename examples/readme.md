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
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples/Merge_Attn/1000.mp4" type="video/mp4">
      </video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 1000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples/Merge_Attn/2000.mp4" type="video/mp4">
      </video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 2000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Merge_Attn/3000.mp4" type="video/mp4">
      </video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 3000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
  </tr>

  <!-- 第二行 -->
  <tr>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Merge_Attn/4000.mp4" type="video/mp4">
      </video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 4000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Merge_Attn/5000.mp4" type="video/mp4">
      </video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 5000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Merge_Attn/6000.mp4" type="video/mp4">
      </video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 6000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
  </tr>

  <!-- 第三行 -->
  <tr>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Merge_Attn/7000.mp4" type="video/mp4">
      </video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 7000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #f6f8fa; border-radius: 8px; text-align: center;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Merge_Attn/8000.mp4" type="video/mp4">
      </video>
      <div style="margin-top: 8px; font-size: 0.95em;">
        <span style="color: #2f855a; font-weight: 600;">🔄 合并 8000 Tokens</span><br>
        <small style="color: #718096;">seq_len:17776</small>
      </div>
    </td>
    <td style="padding: 10px; background: #ebf8ff; border: 2px solid #63b3ed; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
      <video width="100%" style="border-radius: 6px;" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Merge_Attn/ref.mp4" type="video/mp4">
      </video>
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
    <th style="padding: 12px; background: #f0fff4; border-radius: 8px; text-align: center; width: 33%;">
      <div style="font-size: 1.1em; color: #2f855a;">
        🔄 Combined Attention (new method)
        <div style="font-size: 0.9em; color: #718096; margin-top: 4px;">理论计算量下降70-80%</div>
      </div>
    </th>
    <th style="padding: 12px; background: #ebf8ff; border-radius: 8px; text-align: center; width: 34%;">
      <div style="font-size: 1.1em; color: #2b6cb0;">
        ⚖️ 基准对照组
        <div style="font-size: 0.9em; color: #718096; margin-top: 4px;">原始计算量100%</div>
      </div>
    </th>
  </tr>

  <!-- 视频对比行 -->
  <tr valign="top">
    <!-- Smart Attention -->
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Smart_Attn/smartattn1.mp4" type="video/mp4">
      </video>
    </td>
    <!-- Combined Attention -->
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Combined_Attn/combined1.mp4" type="video/mp4">
      </video>
    </td>
    <!-- 基准组 -->
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Combined_Attn/ref1.mp4" type="video/mp4">
      </video>
    </td>
  </tr>

  <!-- 重复其他3组视频对比 -->
  <tr valign="top">
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Smart_Attn/smartattn2.mp4" type="video/mp4">
      </video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Combined_Attn/combined2.mp4" type="video/mp4">
      </video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Combined_Attn/ref2.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
  <tr valign="top">
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Smart_Attn/smartattn3.mp4" type="video/mp4">
      </video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Combined_Attn/combined3.mp4" type="video/mp4">
      </video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Combined_Attn/ref3.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
  <tr valign="top">
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Smart_Attn/smartattn4.mp4" type="video/mp4">
      </video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Combined_Attn/combined4.mp4" type="video/mp4">
      </video>
    </td>
    <td style="padding: 10px; background: #f8fafc; border-radius: 8px;">
      <video width="100%" style="border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);" controls muted playsinline>
        <source src="https://raw.githubusercontent.com/xiaolong-li1/SparseAttnFunctions/refs/heads/dev/examples//Combined_Attn/ref4.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>


To view the corresponding prompt words for the gallery, please click [here](prompts.txt)
</div>
