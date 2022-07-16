import torch

from .base import Transform
from .composite import CompositeTransform
from .squeeze import SqueezeTransform
from normflow.splits import Split, EvenSplit
from normflow.common import Invertible


from nflows.flows import Flow


class GraphNode(Invertible):
    """
    An individual GraphTransform Node
    Encompasses
    1. Merging Operation for Input from Multiple Parent Nodes
    2. Any Arbitrary Inner Transform
    3. Splitting Operation for Output to Multiple Child Nodes
    """
    def __init__(self, inner: Transform, merge: Invertible, split: Invertible):
        """
        @param inner: The inner Transform
        @param merge: The merge operation.
            In the `forward()` computation, takes a tensor `merged_data`
            and returns a dictionary `{parent_nodes: split_data}`
            In the `inverse()` computation, takes a dictionary `{parent_node: split_data}`
            and returns `merged_data`
        @param split: The split operation.
            In the `forward()` computation, takes a tensor `merged_data`
            and returns a dictionary `{child_nodes: split_data}`
            In the `inverse()` computation, takes a dictionary `{child_node: split_data}`
            and returns the `merged_data`
        """
        super().__init__()

        self.inner = inner
        self.merge = merge
        self.split = split

        self.child_nodes = set()
        self.parent_nodes = set()

        self.inputs = {}
        self.outputs = {}

    def connect(self, *children: "GraphNode"):
        """ Connect a Child Node to this GraphNode """
        self.child_nodes.update(set(children))
        for child in children:
            child.parent_nodes.add(self)

    # TODO: logabsdet, adjust merge and split, finish into GraphTransform

    def forward(self):
        """ Perform the forward computation once all inputs are collected """
        self._check_need_have(
            need=self.parent_nodes,
            have=set(self.inputs.keys()),
            extra_msg=f"Node {self} received input from the following unlinked nodes: ",
            missing_msg=f"Node {self} is missing input from the following parent nodes: "
        )

        x = self.merge.inverse(self.inputs)
        z = self.inner.forward(x)
        self.outputs = self.split.forward(z)

        self._check_need_have(
            need=self.child_nodes,
            have=set(self.outputs.keys()),
            extra_msg=f"Node {self} received output for the following unlinked nodes: ",
            missing_msg=f"Node {self} is missing output for the following child nodes: "
        )

        for child in self.child_nodes:
            child.add_inputs(self.outputs[child])

        self.inputs = []

    def inverse(self):
        """ Perform the inverse computation once all outputs are collected """
        self._check_need_have(
            need=self.child_nodes,
            have=set(self.outputs.keys()),
            extra_msg=f"Inverse Node {self} received input from the following unlinked nodes: ",
            missing_msg=f"Inverse Node {self} is missing input from the following child nodes: "
        )
        z = self.split.inverse(self.outputs)
        x = self.inner.inverse(z)
        self.inputs = self.merge.forward(x)

        self._check_need_have(
            need=self.parent_nodes,
            have=set(self.inputs.keys()),
            extra_msg=f"Inverse Node {self} received output from the following unlinked nodes: ",
            missing_msg=f"Inverse Node {self} is missing output for the following parent nodes: "
        )

        for parent in self.parent_nodes:
            parent.add_outputs(self.outputs[parent])

        self.outputs = []

    def add_inputs(self, *inputs: torch.Tensor):
        """ Add one or more input tensors to this GraphNode """
        self.inputs.extend(inputs)

    def add_outputs(self, *outputs: torch.Tensor):
        """ Add one or more output tensors to this GraphNode """
        self.outputs.extend(outputs)

    def _check_need_have(self, need: set, have: set, extra_msg: str = "Extra:", missing_msg: str = "Missing:"):
        """ Check if the items in `need` and `have` are the same and raise appropriate warnings and errors """
        extra = have - need
        if extra:
            raise RuntimeWarning(extra_msg + str(extra))

        missing = need - have
        if missing:
            raise RuntimeError(missing_msg + str(missing))


class GraphNode(Invertible):
    """
    An individual GraphTransform Node
    Encompasses
    1. An Optional Merging Operation for Input from Multiple Parent Nodes
    2. An Inner Transform
    3. An Optional Splitting Operation for Output to Multiple Child Nodes
    """
    def __init__(self, inner: Transform):
        super().__init__()
        self.inner = inner
        self.magic = ...

        self.parent_nodes = set()
        self.child_nodes = set()

        self.inputs = {}
        self.outputs = {}

    @property
    def is_root(self):
        return bool(self.parent_nodes)

    @property
    def is_leaf(self):
        return bool(self.child_nodes)

    def connect(self, *children: "GraphNode"):
        """ Connect one or more children to this node """
        self.child_nodes.update(set(children))
        for child in children:
            child.parent_nodes.add(self)

    def forward(self, x: torch.Tensor, node: "GraphNode"):
        self.inputs[node] = x


    # TODO:
    #  1. How to combine logabsdet?
    #  2. How to determine which part goes to which node? set is not ordered.
    #  3. How to avoid collecting empty outputs? which node returns the output (input or output node)?
    #  4. should this really be in forward() or some other method (collect_input)?
    #  5. should the node save its output to an attribute when it's ready? how does it notify its children of readiness?
    #  6. should the whole output be passed to every child? this would make it a traditional computational graph
    #  7. (related to 6.) maybe this is better done with masking instead of split() and merge()
    #     but masking could become quite complex for large graphs (so should not be left to the end-user)
    #  8. How to handle multiple leaf nodes?

    def forward(self, x: torch.Tensor):
        self.inputs.append(x)

        if len(self.inputs) >= len(self.parent_nodes):

            self.output = self.inner.forward()

    def forward(self, x: torch.Tensor):
        # collect input
        self._inputs.append(x)

        if not self.is_root:
            if len(self._inputs) < len(self.parent_nodes):
                # not enough inputs yet, wait for more
                return

        x = self.merge.inverse(tuple(self._inputs))
        x, logabsdet = self.inner.forward(x)

        parts = self.split.forward(x)

        outputs = []
        # TODO: set is not ordered (so this could be very wrong)
        for part, child in zip(parts, self.child_nodes):
            child.forward(part)



    def inverse(self, *args, **kwargs):
        pass

    def collect_input(self):


class GraphTransform(Transform):
    """
    A Graph Transform combines several individual transforms into any arbitrary computational graph
    """
    def __init__(self, input_node: GraphNode, output_node: GraphNode):
        super().__init__()
        self.input_node = input_node
        self.output_node = output_node

    def forward(self, *args, **kwargs):
        self.input_node.forward(*args, **kwargs)
        return self.output_node.output

    def inverse(self, *args, **kwargs):
        self.output_node.inverse(*args, **kwargs)
        return self.input_node.inverse_output

t0 = GraphNode(...)
t1 = GraphNode(...)
t2 = GraphNode(...)
t3 = GraphNode(...)
t4 = GraphNode(...)
t5 = GraphNode(...)

t0.connect(t5, t2, t1)
t1.connect(t2, t3)
t2.connect(t4)
t3.connect(t4, t5)
t4.connect(t5)


transform = GraphTransform(t0, t5)

x = torch.zeros(0)
z, logabsdet = transform.forward(x)


r"""

   ------------------------------
  /                               \
t0 -------- t2 --------- t4 ------ t5 \
   \       /            /        /
    \     /            /        /
     \   /            /        /
      t1 --------- t3 --------

"""




class MultiScaleCompositeTransform(Transform):




class MultiScaleCompositeTransform(CompositeTransform):
    """
    A multiscale composite transform is a composite transform that
    uses a divide-and-conquer scheme, reshaping ("squeezing") the input and
    passing half of the output of each transform to the next transform

    Introduced by arXiv:1605.08803
    """

    def __init__(self, *transforms: Transform, split_dim: int):
        super().__init__(*transforms)
        self.squeeze = SqueezeTransform()
        self.split = EvenSplit(dim=split_dim)

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        logabsdet = x.new_zeros(x.shape[0])
        for transform in self.transforms:
            x, _ = self.squeeze.forward(x)
            x1, x2 = self.split.forward(x)
            outputs.append(x1)
            print(x1.shape, transform.__class__.__name__, x2.shape)
            x, det = transform.forward(x2)
            print(x.shape)
            logabsdet += det

        print([o.shape for o in outputs])
        outputs = [self.squeeze.inverse(o)[0] for o in outputs]
        print([o.shape for o in outputs])
        z = torch.cat(outputs, dim=self.split.dim)

        print("MultiScale In:", x.shape)
        print("MultiScale Out:", z.shape, logabsdet.shape)

        return z, logabsdet

    def inverse(self, z: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        logabsdets = []
        for transform in reversed(self.transforms):
            z1, z2 = self.split.forward(z)
            outputs.append(z1)
            z, det = transform.inverse(z2)
            logabsdets.append(det)

        x = torch.cat(outputs, dim=self.split.dim)
        logabsdet = torch.stack(logabsdets, dim=0)

        print("MultiScale INV In:", z.shape)
        print("MultiScale INV Out:", x.shape, logabsdet.shape)

        return x, logabsdet

    # def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    #     if x.shape[self.split.dim] < 2:
    #         print("BASE CASE")
    #         # can no longer split
    #         z = x
    #         logabsdet = x.new_zeros(x.shape[0])
    #         return z, logabsdet
    #
    #     # split
    #     x1, x2 = self.split.forward(x)
    #
    #     print("CASCADE CASE", x.shape, x1.shape, x2.shape)
    #
    #     # skip first half
    #     z1 = x1
    #     logabsdet = x.new_zeros(x.shape[0])
    #
    #     # cascade second half
    #     z2, det = super().forward(x2, condition=condition)
    #     logabsdet[-len(det):] += det
    #
    #     # repeat on both halves
    #     # TODO: check if this works well (as it's not the same as in the paper)
    #     z1, det1 = self.forward(z1, condition=condition)
    #     z2, det2 = self.forward(z2, condition=condition)
    #
    #     logabsdet[:len(det1)] += det1
    #     logabsdet[-len(det2):] += det2
    #
    #     # merge
    #     z = self.split.inverse((z1, z2))
    #
    #     return z, logabsdet

    # def inverse(self, z: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    #     # split
    #     z1, z2 = self.split.forward(z)
    #
    #     logabsdet = z.new_zeros(z.shape[0])
    #
    #     # inverse repeat both halves
    #     z1, det1 = self.inverse(z1, condition=condition)
    #     z2, det2 = self.inverse(z2, condition=condition)
    #
    #     logabsdet[:len(det1)] += det1
    #     logabsdet[-len(det2):] += det2
    #
    #     # inverse cascade second half
    #     x2, det = super().inverse(z2, condition=condition)
    #     logabsdet[-len(det):] += det
    #
    #     # inverse skip first half
    #     x1 = z1
    #
    #     # merge
    #     x = self.split.inverse((x1, x2))
    #
    #     return x, logabsdet
