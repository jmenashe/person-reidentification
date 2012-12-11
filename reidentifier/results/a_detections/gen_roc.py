#!/usr/bin/env python

import csv, sys
from os.path import splitext, dirname

def readFile(file):
  data = csv.reader(open(file))
  # Read the column names from the first line of the file
  fields = data.next()
  results = []
  for row in data:
        # Zip together the field names and values
    items = zip(fields, row)
    item = {}
        # Add the value to our dictionary
    for (name, value) in items:
      item[name] = value.strip()
    result = float(item['result'])
    label = int(item['label'])
    results += [[result, label]]
  return results

def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

def getStats(results, threshold = 0.0):
  fp = 0
  tp = 0
  fn = 0
  tn = 0

  for (result, label) in results:
    if result > threshold and label == 1:
      tp += 1
    if result <= threshold and label == 1:
      fn += 1
    if result <= threshold and label == 0:
      tn += 1
    if result > threshold and label == 0:
      fp += 1

  return tp, fp, tn, fn

def getAPR(results, threshold = 0):
  tp, fp, tn, fn = getStats(results,threshold)
  if tp == 0 and fp == 0:
    precision = 0
  else:
    precision = float(tp) / (tp + fp)
  recall = float(tp) / (tp + fn)
  accuracy = float(tp + tn) / (tp + tn + fp + fn)

  return accuracy, precision, recall

def ROC(results, t):
  tp, fp, tn, fn = getStats(results, t)
  tpr = float(tp) / (tp + fn)
  fpr = float(fp) / (fp + tn)
  return fpr, tpr

def PR(results, t):
  tp, fp, tn, fn = getStats(results, t)
  p = float(tp) / (tp + fp)
  r = float(tp) / (tp + fn)
  return r, p

def getBestThreshold(results):
  maxResult = max(map(lambda x: x[0], results))
  minResult = min(map(lambda x: x[0], results))

  r = maxResult - minResult
  step = r / 100.0
  score = 0.0
  threshold = 0.0
  for t in drange(minResult,maxResult,step):
    a,p,r = getAPR(results,t)
    s = 2.0 * p + r
    if score < s:
      score = s
      threshold = t
  return threshold
  
def getCurve(results, fn):
  maxResult = max(map(lambda x: x[0], results))
  minResult = min(map(lambda x: x[0], results))

  r = maxResult - minResult
  step = r / 100.0
  rates = []
  for t in drange(minResult,maxResult,step):
    x, y = fn(results, t)
    rates += [[x, y]]
  return rates

class GraphParams:
  def __init__(self, title = "", ylabel = "True Positive Rate", xlabel = "False Positive Rate"):
    self.title = title
    self.ylabel = ylabel
    self.xlabel = xlabel

def generateCurves(files, params = GraphParams()):
  curves = open("curves.gp", 'w')
  curves.write('set xrange [0:1]; set yrange [0:1];\n')
  curves.write('set xlabel "%s";\n' % params.xlabel)
  curves.write('set ylabel "%s";\n' % params.ylabel)
  curves.write('set title "%s";\n' % params.title)
  curves.write('set datafile separator ",";\n')
  curves.write('set key right center outside;\n')
  curves.write('plot \\\n')
  
  i = 1
  for f, t in files:
    results = readFile(f)
    rates = getCurve(results, ROC)
    f = splitext(f)[0]
    outfile = f + "_roc.csv"
    output = open(outfile, 'w')
    for r in rates:
      output.write("%s,%s\n" %  (r[0], r[1]))
    output.close()

    curves.write('  "%s" u 1:2 title "%s" with lines' % (outfile,t))
    if i == len(files):
      curves.write(';\n')
    else:
      curves.write(', \\\n')
    i += 1
    
  curves.write("pause -1")

files = []
#files += [["hasHat_100w_10s_hog_rbf.csv", "HasHat with RBF"]]
#files += [["hasHat_poly3.csv", "HasHat with 3-Poly"]]
files += [["hasHat.csv", "Has Hat"]]
files += [["hasJeans.csv", "Has Jeans"]]
files += [["hasLongHair.csv", "Has Long Hair"]]
files += [["hasLongPants.csv", "Has Long Pants"]]
files += [["hasLongSleeves.csv", "Has Long Sleeves"]]
files += [["hasShorts.csv", "Has Shorts"]]
files += [["hasTShirt.csv", "Has T-Shirt"]]
files += [["isMale.csv", "Is Male"]]
files += [["hasGlasses.csv", "Has Glasses"]]
for f in files:
  f[0] = "rbf_oneclass_100w_25s/" + f[0]

generateCurves(files, GraphParams(title = "Performance with RBF Kernels"))

for f in files:
  results = readFile(f[0])
  #t = getBestThreshold(results)
  #print max(map(lambda x: x[0], results))
  #print results
  a, p, r = getAPR(results)
  print "%s: A: %2.2f, P: %2.2f, R: %2.2f" % (f[1],a,p,r)
