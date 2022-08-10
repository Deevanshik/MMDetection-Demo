from unittest import TestCase
from unittest.mock import MagicMock

import torch

from mmdet.structures.bbox.bbox_mode import (_bbox_mode_to_name,
                                             bbox_mode_converters, bbox_modes,
                                             convert_bbox_mode,
                                             convert_mask_to_bbox_mode,
                                             get_bbox_mode, register_bbox_mode,
                                             register_bbox_mode_converter)
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from .utils import ToyBaseBoxes


class TestBboxMode(TestCase):

    def setUp(self):
        self.bbox_modes = bbox_modes.copy()
        self.bbox_mode_converters = bbox_mode_converters.copy()
        self._bbox_mode_to_name = _bbox_mode_to_name.copy()

    def tearDown(self):
        # Clear registered items
        bbox_modes.clear()
        bbox_mode_converters.clear()
        _bbox_mode_to_name.clear()
        # Restore original items
        bbox_modes.update(self.bbox_modes)
        bbox_mode_converters.update(self.bbox_mode_converters)
        _bbox_mode_to_name.update(self._bbox_mode_to_name)

    def test_register_bbox_mode(self):
        # test usage of decorator
        @register_bbox_mode('A')
        class A(ToyBaseBoxes):
            pass

        # test usage of normal function
        class B(ToyBaseBoxes):
            pass

        register_bbox_mode('B', B)

        # register class without inheriting from BaseBoxes
        with self.assertRaises(AssertionError):

            @register_bbox_mode('C')
            class C:
                pass

        # test register registered class
        with self.assertRaises(KeyError):

            @register_bbox_mode('A')
            class AA(ToyBaseBoxes):
                pass

        with self.assertRaises(KeyError):
            register_bbox_mode('BB', B)

        @register_bbox_mode('A', force=True)
        class AAA(ToyBaseBoxes):
            pass

        self.assertIs(bbox_modes['a'], AAA)
        self.assertEqual(_bbox_mode_to_name[AAA], 'a')
        register_bbox_mode('BB', B, force=True)
        self.assertIs(bbox_modes['bb'], B)
        self.assertEqual(_bbox_mode_to_name[B], 'bb')
        self.assertEqual(len(bbox_modes), len(_bbox_mode_to_name))

    def test_register_bbox_mode_converter(self):

        @register_bbox_mode('A')
        class A(ToyBaseBoxes):
            pass

        @register_bbox_mode('B')
        class B(ToyBaseBoxes):
            pass

        @register_bbox_mode('C')
        class C(ToyBaseBoxes):
            pass

        # test usage of decorator
        @register_bbox_mode_converter('A', 'B')
        def converter_A(bboxes):
            return bboxes

        # test usage of normal function
        def converter_B(bboxes):
            return bboxes

        register_bbox_mode_converter('B' 'A', converter_B)

        # register uncallable object
        with self.assertRaises(AssertionError):
            register_bbox_mode_converter('A', 'C', 'uncallable str')

        # test register unregistered bbox mode
        with self.assertRaises(AssertionError):

            @register_bbox_mode_converter('A', 'D')
            def converter_C(bboxes):
                return bboxes

        # test register registered converter
        with self.assertRaises(KeyError):

            @register_bbox_mode_converter('A', 'B')
            def converter_D(bboxes):
                return bboxes

        @register_bbox_mode_converter('A', 'B', force=True)
        def converter_E(bboxes):
            return bboxes

        self.assertIs(bbox_mode_converters['a2b'], converter_E)

    def test_get_bbox_mode(self):

        @register_bbox_mode('A')
        class A(ToyBaseBoxes):
            pass

        mode_name, mode_cls = get_bbox_mode('A')
        self.assertEqual(mode_name, 'a')
        self.assertIs(mode_cls, A)
        mode_name, mode_cls = get_bbox_mode(A)
        self.assertEqual(mode_name, 'a')
        self.assertIs(mode_cls, A)

        # get unregistered mode
        class B(ToyBaseBoxes):
            pass

        with self.assertRaises(AssertionError):
            mode_name, mode_cls = get_bbox_mode('B')
        with self.assertRaises(AssertionError):
            mode_name, mode_cls = get_bbox_mode(B)

    def test_convert_bbox_mode(self):

        @register_bbox_mode('A')
        class A(ToyBaseBoxes):
            pass

        @register_bbox_mode('B')
        class B(ToyBaseBoxes):
            pass

        @register_bbox_mode('C')
        class C(ToyBaseBoxes):
            pass

        converter = MagicMock()
        converter.return_value = torch.rand(3, 4, 4)
        register_bbox_mode_converter('A', 'B', converter)

        bboxes_a = A(torch.rand(3, 4, 4))
        th_bboxes_a = bboxes_a.tensor
        np_bboxes_a = th_bboxes_a.numpy()

        # test convert to mode
        convert_bbox_mode(bboxes_a, dst_mode='B')
        self.assertTrue(converter.called)
        converted_bboxes = convert_bbox_mode(bboxes_a, dst_mode='A')
        self.assertIs(converted_bboxes, bboxes_a)
        # test convert to unregistered mode
        with self.assertRaises(AssertionError):
            convert_bbox_mode(bboxes_a, dst_mode='C')

        # test convert tensor and ndarray
        # without specific src_mode
        with self.assertRaises(AssertionError):
            convert_bbox_mode(th_bboxes_a, dst_mode='B')
        with self.assertRaises(AssertionError):
            convert_bbox_mode(np_bboxes_a, dst_mode='B')
        # test np.ndarray
        convert_bbox_mode(np_bboxes_a, src_mode='A', dst_mode='B')
        converted_bboxes = convert_bbox_mode(
            np_bboxes_a, src_mode='A', dst_mode='A')
        self.assertIs(converted_bboxes, np_bboxes_a)
        # test tensor
        convert_bbox_mode(th_bboxes_a, src_mode='A', dst_mode='B')
        converted_bboxes = convert_bbox_mode(
            th_bboxes_a, src_mode='A', dst_mode='A')
        self.assertIs(converted_bboxes, th_bboxes_a)
        # test other type
        with self.assertRaises(TypeError):
            convert_bbox_mode([[1, 2, 3, 4]], src_mode='A', dst_mode='B')

    def test_convert_mask_to_box_mode(self):

        @register_bbox_mode('A')
        class A(ToyBaseBoxes):
            pass

        A.from_bitmap_masks = MagicMock()
        A.from_polygon_masks = MagicMock()

        bitmap_masks = BitmapMasks.random()
        convert_mask_to_bbox_mode(bitmap_masks, A)
        self.assertTrue(A.from_bitmap_masks.called)
        polygon_masks = PolygonMasks.random()
        convert_mask_to_bbox_mode(polygon_masks, A)
        self.assertTrue(A.from_polygon_masks.called)
