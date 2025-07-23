# 🧠 Zero-Shot Medical Anomaly Detection using CLIP

This repository contains a hands-on implementation and analysis inspired by the CVPR 2024 paper:

**Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images**  
[📄 Paper Link (CVPR 2024)](https://arxiv.org/abs/2403.12570)

---

## 📌 Project Summary

This project demonstrates a **training-free anomaly detection** approach using **OpenCLIP**, tested on a **public Brain MRI dataset**. It replicates the core idea of the paper: detecting abnormal regions in medical images by leveraging **semantic distances in CLIP's feature space** — **without any fine-tuning**.

---

## 🧪 What’s Implemented

- ✅ Load Brain MRI dataset (Normal & Tumor scans)
- ✅ Extract CLIP features using OpenCLIP
- ✅ Calculate **cosine distance** between test images and reference (normal) images
- ✅ Visualize the anomaly detection result with **histogram comparison**
- ✅ Threshold-based anomaly flagging

---

## 🔍 Sample Output

![Cosine Distance Histogram](outputs/histogram.png)

In this figure:
- **Blue bars** represent cosine distances of normal test images to reference normals
- **Orange bars** represent tumor images’ distances
- Tumor images clearly deviate from the normal cluster

---

## 🧠 About the Paper

The paper proposes a **multi-modal, multi-level adaptation framework** that:

- Uses **frozen CLIP encoders**
- Trains lightweight **visual adapters** across MRI, CT, and X-ray modalities
- Supports **zero-shot and few-shot branches** via a shared visual memory and text guidance

It significantly outperforms prior methods like BGAD, DRA, and April-GAN on 6 medical datasets.

---

## 📁 Folder Structure

