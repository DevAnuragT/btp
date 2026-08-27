#import "@preview/touying:0.5.3": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Model Architecture],
    subtitle: [Dual-Stream Inventory Shortcut LSTM],
    author: [Anurag & Team],
    date: datetime.today(),
    institution: [IIITM],
  ),
)

#title-slide()

== Architecture: Dual-Stream Inventory Shortcut

#align(center)[
  #v(2em)

  // Upload 'architecture.png' to your Typst web app project to render it here!
  #image("architecture.png", width: 95%)
]

== Key Architectural Innovations

To solve the vanishing signal problem and isolate causal factors, we designed a custom LSTM architecture that splits the data and implements a bypass mechanism.

#v(1em)

*1. Dual-Stream Processing:* 
We separate the 14-day history into two branches:
- *Demand:* Sales, discounts, holidays.
- *Inventory:* Daily stock status.

This prevents erratic daily sales noise from washing out the strict physical reality of stock levels.

#v(1em)

*2. The Inventory Shortcut:*
Even LSTMs forget over a 14-day window. The single most important predictor of tomorrow's stockout is *today's* inventory. We physically bypass the recurrent layers and feed today's inventory straight into the final classifier.
