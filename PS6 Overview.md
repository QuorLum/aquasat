**Core Objective**

* we need to identify and map crop types for the current season using multi-temporal spectral signatures.
* The system must track the specific growth/phenological stages of the crops and detect moisture stress levels dynamically using combined optical and radar inputs.
* We must translate calculated crop water deficits into actionable, pixel-level irrigation advisory maps. The study area will focus on a canal command area that includes a mix of both irrigated and tail-end rainfed regions.



**Professor mentioned** 

* Optical Satellite Imagery : LISS-III, Sentinel-2 :: Tracking crop signatures and vegetation indices during clear sky conditions.
* Microwave (Radar) Data : EOS-04 (Risat-1), Sentinel-1 :: All-weather, cloud-penetrating imagery to maintain continuous tracking during rainy cycles.



**A baseline crop classification accuracy must be of greater than 80% to 85%.**



**## *autonomous drone integration for ground validation is not expected or required for this project.***



we can use Google Earth Engine through it's API , and for crop classification we can train a temporal model like LSTM or a 1D-CNN on the time-series spectral signature to classify the crop type . Once the crop is identified, the model needs to determine its growth stage using vegetation indices (like NDVI) and radar backscatter over time.



The problem I noticed while analysing google map that the image of Indian land like crops buildings wasn't clear enough to precisely identify anything .

i found out that it is a common issue, Google Earth relies entirely on optical imagery, which is often degraded over India due to local security regulations, or obscured by seasonal cloud cover.
mentors will provide us almost raw data of satellite and we have to use it wisely .



one more problem i came up with that if we use multiple satellite data at once then how will that gonna collaborate with exact same time because if we feed any data which doesn't match with exact time then we for sure to get wrong output.

and i found out that It is one of the biggest headaches in remote sensing. Sentinel-2 (optical) might pass over a farm every 5 days, while Sentinel-1 (radar) might pass over every 6 to 12 days. They almost never snap a picture at the exact same second. if we feed misaligned timestamps into a machine learning model, it will look like chaotic noise.

**The solution :** Agriculture is slow process , we can sync data in Temporal Windows 

&#x09;Instead of using raw daily images, divide the crop season into standard intervals—say, 10-day or 15-day "bins" (e.g., June 1-10, June 11-20).



During the monsoon season, our 10-day optical bin might be 100% clouds. we cannot feed clouds to our model. 

SO we can use a mathematical technique called Interpolation (like linear interpolation or a Savitzky-Golay filter). If we have clear optical data in June and clear optical data in August, the algorithm mathematically connects the dots for the cloudy July gap.



Meanwhile, our Radar data (which penetrates clouds) remains uninterrupted, acting as the "anchor" that tells the model exactly what is happening on the ground while the optical sensors are blind.



**Problem :** one of the biggest limitations in traditional remote sensing. If a severe heatwave or sudden dry spell hits on day two of a ten-day window, waiting for the next satellite pass to tell the farmer their crop is dying is completely useless.

**Solution :** we can use The Two-Speed Architecture 

1. Slow Track : 10-day satellite method works perfectly because recently planted rice doesn't suddenly turn into wheat overnight . 
2. Fast Track : For the irrigation advisory, we abandon the slow satellite wait time. Instead, we build a predictive model  that calculates the daily probability of moisture stress.

&#x20;

And Instead of relying solely on today's blurry or missing satellite image, we train an algorithm (like an LSTM network, which is great for time-series data, or an XGBoost model) on massive amounts of historical data.

We feed the AI 5 to 10 years of historical data for the pilot region.

* Input : Historical daily temperatures, historical rainfall, wind speed, the crop's growth stage, and the historical satellite radar (SAR) readings.
* Output : The actual historical moisture stress levels (often measured by historical ground truth or derived indices like the NDWI - Normalized Difference Water Index).



* The AI learns the complex, hidden relationships. It learns that: "If the crop is in week 4 of its growth stage, and the temperature has been above 38°C for three days with zero rainfall, there is a 92% probability of severe moisture stress—even if the satellite hasn't taken a new picture yet."
* In production, the system uses the latest available satellite image as a "baseline." From that day forward, it ingests daily, real-time meteorological data (which updates every few hours, unlike satellites) to constantly forecast the current and tomorrow's moisture stress probability. 
* we can also analyze real time wind speed and direction to calculate upcoming catastrophe , and we will do that by analysing whole weather of India instead of only one spot or we can analyse weather in the radius of 200 - 500 KM radius for accurate and faster outcomes . 





I don't know but I guess it can predict crop demand in real time with different types of satellite data like infrared or many types of frequency emitted by crops .  in real life situation we can take help from other countries satellite which is passing by our area instead of waiting days our satellite to return and get us the data . 



I found out that When a crop gets thirsty, its physical and chemical properties change, and it reacts differently across the electromagnetic spectrum:

* Near-Infrared (NIR) \& Shortwave Infrared (SWIR): Healthy plants reflect a massive amount of NIR light because of the water structure inside their leaves. When they dry out, that leaf structure collapses, and they start absorbing more SWIR light. By calculating the ratio between these two public bands (known as the Normalized Difference Water Index or NDWI), your AI can directly map leaf moisture.
* Thermal Infrared (Heat Emission): This is exactly what I meant by "emitted frequencies." When plants have plenty of water, they "sweat" (transpire) to stay cool. When they run out of water, they stop sweating, and their canopy temperature shoots up. Thermal satellites pick up this emitted heat signature, signaling a moisture crisis days before the plant physically turns brown.
* Space agencies around the world practice an open-data policy for Earth observation. When we look at the problem statement details, we'll see they explicitly paired Indian satellites (LISS-III and EOS-04) with international ones: Sentinel-1 and Sentinel-2, which belong to the European Space Agency (ESA).By combining India's satellites with Europe's satellites, we create what the industry calls a Virtual Constellation.
* Instead of waiting 10–12 days for a single Indian satellite to return, we weave the data tracks together. If an ESA satellite passes over Bihar on Tuesday, and an ISRO satellite passes over on Thursday, our predictive AI model ingests both. This dramatically cuts down our wait time and gives our real-time model a continuous stream of fresh data points.



**Some Satellite Details** 

1. Sentinel-2 , Agency: ESA(Europe) , Type of data collected : optical/multispectral , Revisit Time : \~5 days 
2. LISS-III , Agency : ISRO(India) , Type of data collected : optical/multispectral , Revisit Time : \~24 days
3. Sentinal-1 , Agency : ESA(Europe) , type of data collected : C-band Radar(SAR) , revisit time :\~6-12 days
4. EOS-04(Risat-1) , Agency : ISRO(India) , type of data collected : S-band radar(SAR) , revisit time \~14 days



Satellite data alone isn't enough. we need ground truth to anchor the model. The mentors explicitly stated they will provide the meteorological datasets (like daily gridded rainfall and temperature) alongside soil maps and canal boundaries. our AI will use this ground data as a primary input feature to compute immediate daily evapotranspiration and water deficits.

