#import "@preview/touying:0.5.3": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Predicting Store-Level Stockouts in Fresh Retail],
    subtitle: [B.Tech Project Problem Statement],
    author: [Anurag & Team],
    date: datetime.today(),
    institution: [IIITM],
  ),
)

#title-slide()

== The Problem Statement

Fresh retail products (produce, meat, dairy) have extremely short shelf lives, creating a tightrope walk for store managers:

- #alert[*Overstocking*] leads to massive spoilage, financial loss, and food waste.
- #alert[*Understocking*] leads to **stockouts** (empty shelves), lost immediate sales, and frustrated customers.

*The Challenge:* 
Traditional supply chain forecasting operates at the warehouse level over weekly horizons. However, managing fresh inventory requires **store-level, hourly precision**.

*Our Objective:*
To build a Deep Learning architecture capable of predicting the **exact hourly stockout status** of a product within a specific store for the next 24 hours. This allows managers to proactively trigger intra-day replenishment and minimize both waste and lost sales.
