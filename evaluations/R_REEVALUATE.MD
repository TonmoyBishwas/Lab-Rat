## Q1 — EASY · Line chart of one month of daily temperature

Using `ggplot2`, plot a line chart of daily temperature across one month
(30 days). Generate synthetic data: dates from `2026-06-01` to `2026-06-30`,
temperatures around 30 °C with `rnorm` noise. Add points on top of the line.
Title the plot, label the axes, format the x-axis dates as `Jun 01`, `Jun 05`,
etc.


Assumptions:
- The data contains two columns: `date` (Date class) and `temp` (numeric).
- The data spans 30 days, from 2026-06-01 to 2026-06-30.
- The temperature values are centered around 30 °C.
- The plot type is a line chart with overlaid points.

```r
library(ggplot2)
library(lubridate)
library(dplyr)

# 1. Generate synthetic data
start_date <- ymd("2026-06-01")
end_date <- ymd("2026-06-30")
dates <- seq(start_date, end_date, by = "day")

# Generate temperatures around 30 degrees C with noise
set.seed(42) # for reproducibility
temperatures <- rnorm(length(dates), mean = 30, sd = 3)

# Create the data frame
temp_data <- data.frame(date = dates, temp = temperatures)

# 2. Plotting
p <- ggplot(temp_data, aes(x = date, y = temp)) +
  # Add the line chart
  geom_line(color = "steelblue", linewidth = 0.8) +
  # Add points on top of the line
  geom_point(size = 2, color = "darkred") +
  # Format the x-axis dates as "Jun 01"
  scale_x_date(date_labels = "%b %d") +
  # Add labels and title
  labs(
    title = "Daily Temperature Across June 2026",
    x = "Date",
    y = "Temperature (°C)"
  ) +
  theme_minimal()

print(p)
```

Notes:
The `scale_x_date(date_labels = "%b %d")` function is crucial for achieving the specific date formatting requested ("Jun 01").
`geom_line` connects the daily points, while `geom_point` ensures the individual data points are visible on the chart.

#OUTPUT: "H:\Documents\RploSDFDFt01.png"
Attaching package: ‘lubridate’

The following objects are masked from ‘package:base’:

    date, intersect, setdiff, union

> library(dplyr)

> # 1. Generate synthetic data
> start_date <- ymd("2026-06-01")

> end_date <- ymd("2026-06-30")

> dates <- seq(start_date, end_date, by = "day")

> # Generate temperatures around 30 degrees C with noise
> set.seed(42) # for reproducibility

> temperatures <- rnorm(length(dates), mean = 30, sd = 3)

> # Create the data frame
> temp_data <- data.frame(date = dates, temp = temperatures)

> # 2. Plotting
> p <- ggplot(temp_data, aes(x = date, y = temp)) +
+   # Add the line chart
+   geom_line(color = "steelblue", linewidth = 0.8) +
+   .... [TRUNCATED] 

> print(p)


## Q6 — HARD · Vectorised FizzBuzz, 1 through 30

Print the FizzBuzz sequence from 1 to 30:

- Multiples of 15 print `"FizzBuzz"`
- Multiples of 3 (but not 5) print `"Fizz"`
- Multiples of 5 (but not 3) print `"Buzz"`
- Everything else prints the number itself

**Do not use a for-loop.** Use only vectorised operations (`ifelse` or
`case_when`) and `%%`.



Assumptions:
- The input is a sequence of integers from 1 to 30.
- The output must be a character vector containing the FizzBuzz results.
- `case_when` is the preferred vectorized method.

```r
library(dplyr)

# Create the vector of numbers
x <- 1:30

# Apply FizzBuzz logic using case_when
result <- case_when(
  x %% 15 == 0 ~ "FizzBuzz",
  x %% 3  == 0 ~ "Fizz",
  x %% 5  == 0 ~ "Buzz",
  TRUE     ~ as.character(x)
)

print(result)
```

Notes:
The order of conditions in `case_when` is crucial; checking for multiples of 15 first ensures that numbers like 15 and 30 are correctly labeled "FizzBuzz" and not just "Fizz" or "Buzz". Using `as.character(x)` in the final `TRUE` condition ensures all non-matching numbers are converted to strings for consistent output.


#OUTPUT:
 # Apply FizzBuzz logic using case_when
> result <- case_when(
+   x %% 15 == 0 ~ "FizzBuzz",
+   x %% 3  == 0 ~ "Fizz",
+   x %% 5  == 0 ~ "Buzz",
+ .... [TRUNCATED] 

> print(result)
 [1] "1"        "2"        "Fizz"     "4"        "Buzz"     "Fizz"     "7"        "8"        "Fizz"     "Buzz"     "11"      
[12] "Fizz"     "13"       "14"       "FizzBuzz" "16"       "17"       "Fizz"     "19"       "Buzz"     "Fizz"     "22"      
[23] "23"       "Fizz"     "Buzz"     "26"       "Fizz"     "28"       "29"       "FizzBuzz"