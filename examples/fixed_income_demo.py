import larp_quantmath as lq

face_value = 1000
coupon_rate = 0.05
yield_rate = 0.04
years = 5
payments_per_year = 1

price = lq.bond_price(face_value, coupon_rate, yield_rate, years, payments_per_year)
duration = lq.macaulay_duration(face_value, coupon_rate, yield_rate, years, payments_per_year)
modified_duration = lq.modified_duration(face_value, coupon_rate, yield_rate, years, payments_per_year)
convexity = lq.bond_convexity(face_value, coupon_rate, yield_rate, years, payments_per_year)
dv01 = lq.dv01(face_value, coupon_rate, yield_rate, years, payments_per_year)

print("Bond price:", price)
print("Macaulay duration:", duration)
print("Modified duration:", modified_duration)
print("Convexity:", convexity)
print("DV01:", dv01)