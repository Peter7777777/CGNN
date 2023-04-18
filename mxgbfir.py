import sys
import re

_comparer = None


def FeatureScoreComparer(sortingMetric):
    global _comparer
    _comparer = {
        'gain': lambda x: -x.Gain,
        'cover': lambda x: -x.Cover,
        'fscore': lambda x: -x.FScore,
        'fscoreweighted': lambda x: -x.FScoreWeighted,
        'averagefscoreweighted': lambda x: -x.AverageFScoreWeighted,
        'averagegain': lambda x: -x.AverageGain,
        'averagecover': lambda x: -x.AverageCover,
        'expectedgain': lambda x: -x.ExpectedGain
    }[sortingMetric.lower()]


class SplitValueHistogram:
    def __init__(self):
        self.values = {}

    def AddValue(self, splitValue, count):
        if not (splitValue in self.values):
            self.values[splitValue] = 0
        self.values[splitValue] += count

    def Merge(self, histogram):
        for key in histogram.values.keys():
            self.AddValue(key, histogram.values[key])


class FeatureInteraction:
    def __init__(self, interaction, gain, cover, pathProbability, depth, treeIndex, fScore=1):
        self.SplitValueHistogram = SplitValueHistogram()

        features = sorted(interaction, key=lambda x: x.Feature)
        self.Name = "|".join(x.Feature for x in features)
        self.Depth = len(interaction) - 1
        self.Gain = gain
        self.Cover = cover
        self.FScore = fScore
        self.FScoreWeighted = pathProbability
        self.AverageFScoreWeighted = self.FScoreWeighted / self.FScore
        self.AverageGain = self.Gain / self.FScore
        self.AverageCover = self.Cover / self.FScore
        self.ExpectedGain = self.Gain * pathProbability
        self.TreeIndex = treeIndex
        self.TreeDepth = depth
        self.AverageTreeIndex = self.TreeIndex / self.FScore
        self.AverageTreeDepth = self.TreeDepth / self.FScore
        self.HasLeafStatistics = False

        if self.Depth == 0:
            self.SplitValueHistogram.AddValue(interaction[0].SplitValue, 1)

        self.SumLeafValuesLeft = 0.0
        self.SumLeafCoversLeft = 0.0
        self.SumLeafValuesRight = 0.0
        self.SumLeafCoversRight = 0.0

    def __lt__(self, other):
        return self.Name < other.Name


class FeatureInteractions:
    def __init__(self):
        self.Count = 0
        self.interactions = {}

    def GetFeatureInteractionsOfDepth(self, depth):
        return sorted([self.interactions[key] for key in self.interactions.keys() if self.interactions[key].Depth == depth], key=_comparer)

    def GetFeatureInteractionsWithLeafStatistics(self):
        return sorted([self.interactions[key] for key in self.interactions.keys() if self.interactions[key].HasLeafStatistics], key=_comparer)

    def Merge(self, other):
        for key in other.interactions.keys():
            fi = other.interactions[key]
            if not (key in self.interactions):
                self.interactions[key] = fi
            else:
                self.interactions[key].Gain += fi.Gain
                self.interactions[key].Cover += fi.Cover
                self.interactions[key].FScore += fi.FScore
                self.interactions[key].FScoreWeighted += fi.FScoreWeighted
                self.interactions[key].AverageFScoreWeighted = self.interactions[key].FScoreWeighted / self.interactions[key].FScore
                self.interactions[key].AverageGain = self.interactions[key].Gain / self.interactions[key].FScore
                self.interactions[key].AverageCover = self.interactions[key].Cover / self.interactions[key].FScore
                self.interactions[key].ExpectedGain += fi.ExpectedGain
                self.interactions[key].SumLeafCoversLeft += fi.SumLeafCoversLeft
                self.interactions[key].SumLeafCoversRight += fi.SumLeafCoversRight
                self.interactions[key].SumLeafValuesLeft += fi.SumLeafValuesLeft
                self.interactions[key].SumLeafValuesRight += fi.SumLeafValuesRight
                self.interactions[key].TreeIndex += fi.TreeIndex
                self.interactions[key].AverageTreeIndex = self.interactions[key].TreeIndex / self.interactions[key].FScore
                self.interactions[key].TreeDepth += fi.TreeDepth
                self.interactions[key].AverageTreeDepth = self.interactions[key].TreeDepth / self.interactions[key].FScore
                self.interactions[key].SplitValueHistogram.Merge(fi.SplitValueHistogram)


class XgbModel:
    def __init__(self, verbosity=0):
        self._verbosity = verbosity
        self.XgbTrees = []
        self._treeIndex = 0
        self._maxDeepening = 0
        self._pathMemo = []
        self._maxInteractionDepth = 0

    def AddTree(self, tree):
        self.XgbTrees.append(tree)

    def GetFeatureInteractions(self, maxInteractionDepth, maxDeepening):
        xgbFeatureInteractions = FeatureInteractions()
        self._maxInteractionDepth = maxInteractionDepth
        self._maxDeepening = maxDeepening

        if self._verbosity >= 1:
            if self._maxInteractionDepth == -1:
                print("Collectiong feature interactions")
            else:
                print("Collectiong feature interactions up to depth {}".format(self._maxInteractionDepth))

        for i, tree in enumerate(self.XgbTrees):
            if self._verbosity >= 2:
                sys.stdout.write("Collectiong feature interactions within tree #{} ".format(i + 1))

            self._treeFeatureInteractions = FeatureInteractions()
            self._pathMemo = []
            self._treeIndex = i

            treeNodes = []
            self.CollectFeatureInteractions(tree, treeNodes, currentGain=0.0, currentCover=0.0, pathProbability=1.0, depth=0, deepening=0)

            if self._verbosity >= 2:
                sys.stdout.write("=> number of interactions: {}\n".format(len(self._treeFeatureInteractions.interactions)))
            xgbFeatureInteractions.Merge(self._treeFeatureInteractions)

        if self._verbosity >= 1:
            print("{} feature interactions has been collected.".format(len(xgbFeatureInteractions.interactions)))

        return xgbFeatureInteractions

    def CollectFeatureInteractions(self, tree, currentInteraction, currentGain, currentCover, pathProbability, depth, deepening):
        if tree.node.IsLeaf:
            return
        currentInteraction.append(tree.node)
        currentGain += tree.node.Gain
        currentCover += tree.node.Cover

        pathProbabilityLeft = pathProbability * (tree.left.node.Cover / tree.node.Cover)
        pathProbabilityRight = pathProbability * (tree.right.node.Cover / tree.node.Cover)

        fi = FeatureInteraction(currentInteraction, currentGain, currentCover, pathProbability, depth, self._treeIndex, 1)

        if (depth < self._maxDeepening) or (self._maxDeepening < 0):
            newInteractionLeft = []
            newInteractionRight = []

            self.CollectFeatureInteractions(tree.left, newInteractionLeft, 0.0, 0.0, pathProbabilityLeft, depth + 1, deepening + 1)
            self.CollectFeatureInteractions(tree.right, newInteractionRight, 0.0, 0.0, pathProbabilityRight, depth + 1, deepening + 1)

        path = ",".join(str(n.Number) for n in currentInteraction)

        if not (fi.Name in self._treeFeatureInteractions.interactions):
            self._treeFeatureInteractions.interactions[fi.Name] = fi
            self._pathMemo.append(path)
        else:
            if path in self._pathMemo:
                return
            self._pathMemo.append(path)

            tfi = self._treeFeatureInteractions.interactions[fi.Name]
            tfi.Gain += currentGain
            tfi.Cover += currentCover
            tfi.FScore += 1
            tfi.FScoreWeighted += pathProbability
            tfi.AverageFScoreWeighted = tfi.FScoreWeighted / tfi.FScore
            tfi.AverageGain = tfi.Gain / tfi.FScore
            tfi.AverageCover = tfi.Cover / tfi.FScore
            tfi.ExpectedGain += currentGain * pathProbability
            tfi.TreeDepth += depth
            tfi.AverageTreeDepth = tfi.TreeDepth / tfi.FScore
            tfi.TreeIndex += self._treeIndex
            tfi.AverageTreeIndex = tfi.TreeIndex / tfi.FScore
            tfi.SplitValueHistogram.Merge(fi.SplitValueHistogram)

        if len(currentInteraction) - 1 == self._maxInteractionDepth:
            return

        currentInteractionLeft = list(currentInteraction)
        currentInteractionRight = list(currentInteraction)

        leftTree = tree.left
        rightTree = tree.right

        if leftTree.node.IsLeaf and (deepening == 0):
            tfi = self._treeFeatureInteractions.interactions[fi.Name]
            tfi.SumLeafValuesLeft += leftTree.node.LeafValue
            tfi.SumLeafCoversLeft += leftTree.node.Cover
            tfi.HasLeafStatistics = True

        if rightTree.node.IsLeaf and (deepening == 0):
            tfi = self._treeFeatureInteractions.interactions[fi.Name]
            tfi.SumLeafValuesRight += rightTree.node.LeafValue
            tfi.SumLeafCoversRight += rightTree.node.Cover
            tfi.HasLeafStatistics = True

        self.CollectFeatureInteractions(tree.left, currentInteractionLeft, currentGain, currentCover, pathProbabilityLeft, depth + 1, deepening)
        self.CollectFeatureInteractions(tree.right, currentInteractionRight, currentGain, currentCover, pathProbabilityRight, depth + 1, deepening)


class XgbTreeNode:
    def __init__(self):
        self.Feature = ''
        self.Gain = 0.0
        self.Cover = 0.0
        self.Number = -1
        self.LeftChild = None
        self.RightChild = None
        self.LeafValue = 0.0
        self.SplitValue = 0.0
        self.IsLeaf = False

    def __lt__(self, other):
        return self.Number < other.Number


class XgbTree:
    def __init__(self, node):
        self.left = None
        self.right = None
        self.node = node


class XgbModelParser:
    def __init__(self, verbosity=0):
        self._verbosity = verbosity
        self.nodeRegex = re.compile(r'(\d+):\[(.*?)(?:<(.+)|)\]\syes=(.*),no=(.*?),(?:missing=.*,)?gain=(.*),cover=(.*)')
        self.leafRegex = re.compile(r'(\d+):leaf=(.*),cover=(.*)')

    def ConstructXgbTree(self, tree):
        if tree.node.LeftChild is not None:
            tree.left = XgbTree(self.xgbNodeList[tree.node.LeftChild])
            self.ConstructXgbTree(tree.left)
        if tree.node.RightChild is not None:
            tree.right = XgbTree(self.xgbNodeList[tree.node.RightChild])
            self.ConstructXgbTree(tree.right)

    def ParseXgbTreeNode(self, line):
        node = XgbTreeNode()

        m = self.leafRegex.match(line)
        if m:
            node.Number = int(m.group(1))
            node.LeafValue = float(m.group(2))
            node.Cover = float(m.group(3))
            node.IsLeaf = True
        else:
            m = self.nodeRegex.match(line)
            node.Number = int(m.group(1))
            node.Feature = m.group(2)
            node.SplitValue = float(m.group(3)) if m.group(3) else 0.5
            node.LeftChild = int(m.group(4))
            node.RightChild = int(m.group(5))
            node.Gain = float(m.group(6))
            node.Cover = float(m.group(7))
            node.IsLeaf = False
        return node

    def GetXgbModelFromMemory(self, dump, maxTrees):
        model = XgbModel(self._verbosity)
        self.xgbNodeList = {}
        numTree = 0
        for booster_line in dump:
            self.xgbNodeList = {}
            for line in booster_line.split('\n'):
                line = line.strip()
                if not line:
                    continue
                node = self.ParseXgbTreeNode(line)
                if not node:
                    return None
                self.xgbNodeList[node.Number] = node
            numTree += 1
            tree = XgbTree(self.xgbNodeList[0])
            self.ConstructXgbTree(tree)
            model.AddTree(tree)
            if numTree == maxTrees:
                break
        return model


def GetStatistics(booster, dump, feature_names=None, MaxTrees=100, MaxInteractionDepth=2, MaxDeepening=-1, SortBy='Gain'):
    if 'get_dump' not in dir(booster):
        if 'get_booster' in dir(booster):
            booster = booster.get_booster()
        elif 'booster' in dir(booster):
            booster = booster.booster()
        else:
            return -20
    if feature_names is not None:
        if isinstance(feature_names, list):
            booster.feature_names = feature_names
        else:
            booster.feature_names = list(feature_names)
    FeatureScoreComparer(SortBy)
    xgbParser = XgbModelParser()
    xgbModel = xgbParser.GetXgbModelFromMemory(dump, MaxTrees)
    featureInteractions = xgbModel.GetFeatureInteractions(MaxInteractionDepth, MaxDeepening)
    interactions_dict = {}
    for i in range(MaxInteractionDepth + 1):
        interactions_dict['Depth' + str(i)] = featureInteractions.GetFeatureInteractionsOfDepth(i)
    return interactions_dict


def GetExpectedGain(Interactions_dict):
    Interaction, ExpectedGain = [], []
    for fi in Interactions_dict['Depth0']:
        Interaction.append(fi.Name)
        ExpectedGain.append(fi.ExpectedGain)
    return Interaction, ExpectedGain


def GetImportanceFeature(Interactions_dict, Depth='Depth1', Index='expectedgain'):
    class FeatureMeta:
        def __init__(self, feature, importanceValue):
            self.feature = feature
            self.importanceValue = importanceValue

    FeatureList = []
    for fi in Interactions_dict[Depth]:
        feature = int(fi.Name[1:]) if fi.Name[0] == 'f' else int(fi.Name)
        if Index.lower() == 'gain':
            importanceValue = fi.Gain
        elif Index.lower() == 'cover':
            importanceValue = fi.Cover
        elif Index.lower() == 'fscore':
            importanceValue = fi.FScore
        elif Index.lower() == 'fscoreweighted':
            importanceValue = fi.FScoreWeighted
        elif Index.lower() == 'averagefscoreweighted':
            importanceValue = fi.AverageFScoreWeighted
        elif Index.lower() == 'averagegain':
            importanceValue = fi.AverageGain
        elif Index.lower() == 'averagecover':
            importanceValue = fi.AverageCover
        elif Index.lower() == 'expectedgain':
            importanceValue = fi.ExpectedGain
        featureMeta = FeatureMeta(feature, importanceValue)
        FeatureList.append(featureMeta)
    return FeatureList


def GetAverageTreeDepth(Interactions_dict, Depth='Depth1'):
    Interaction, AverageTreeDepth = [], []
    for fi in Interactions_dict[Depth]:
        Interaction.append(fi.Name)
        AverageTreeDepth.append(fi.AverageTreeDepth)
    return Interaction, AverageTreeDepth