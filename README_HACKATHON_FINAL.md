# 🌸 BloomWatch  
**NASA MODIS Plant Bloom Detection**  
*NASA Space Apps Challenge 2025 – Global Award Submission*  

---

## 🚀 Why BloomWatch?  
Every spring, flowers bloom across our planet 🌍. But how can we track these changes **from space**?  
BloomWatch answers this with an AI-powered system that detects and classifies plant bloom stages directly from **NASA MODIS satellite data**.  

This project shows how space technology can support:  
- 🌾 **Agriculture** – crop monitoring  
- 🌱 **Climate research** – seasonal vegetation shifts  
- 🐝 **Biodiversity** – pollination and ecosystem health  

---

## 🛰️ The NASA Data We Used  
- **Real MODIS HDF4 files** (2020–2025)  
- Extracted **NDVI & EVI vegetation indices**  
- Multiple global regions, multi-year span  
- ~100 authentic satellite granules  

We combined this with **synthetic augmentation** to build a balanced dataset of **153 samples**.  

---

## 🤖 The BloomWatch AI  
A custom **Convolutional Neural Network (CNN)** trained on vegetation indices:  

- **Input:** 2-channel (NDVI + EVI) images, resized to 224×224  
- **Output:** 5 bloom stages 🌱🌼🌸🌺🍂  
- **Performance:** ~98% accuracy on test data  
- **Lightweight:** 1.7 MB model, runs in seconds  

---

## 🌼 Bloom Stages Classified  
- **Bud** → early growth  
- **Early Bloom** → first flowers  
- **Full Bloom** → peak flowering  
- **Late Bloom** → declining stage  
- **Dormant** → no active growth  

---

## 📊 Results That Matter  
✔️ Accurate across regions and years  
✔️ Compact & deployable model  
✔️ Trained in under 2 minutes  
✔️ Balanced dataset for fairness  

---

## 🌍 Why It Matters  
BloomWatch isn’t just about flowers. It’s about:  

- **Food security** – supporting farmers with satellite-based crop insights  
- **Climate action** – monitoring seasonal cycles under global warming  
- **Conservation** – helping track flowering patterns for pollinators and biodiversity  

---

## 🔬 Innovation Highlights  
✨ Direct NASA HDF4 processing (no pre-processed shortcuts)  
✨ Hybrid training: real + synthetic data  
✨ Small but powerful CNN model  
✨ End-to-end pipeline: from satellite granules → bloom stage predictions  

---

## 📞 Team & Submission  
**Project:** BloomWatch – NASA MODIS Plant Bloom Detection  
**Challenge:** NASA Space Apps 2025 – Global Award  
**Data Source:** NASA Earthdata MODIS Collection  

---

👉 BloomWatch shows how **AI + NASA satellite data** can make planetary-scale bloom monitoring possible.  
Together, let’s watch our planet bloom. 🌸  
