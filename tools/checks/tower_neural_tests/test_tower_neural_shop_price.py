"""Located payable amounts, shared shelf/purchase reading and native ink OCR."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agent"))
from custom.climb_tower_neural.shop_ui.handler import ShopHandler
from custom.climb_tower_neural.shop_ui.interactor import PRICE_NODE, ShopInteractor
from custom.climb_tower_neural.shop_ui.price import payable_amount, price_key
from custom.climb_tower_neural.shop_ui.purchase import ShopItemPrice


ROI = [10, 10, 110, 35]


def row(text, box=(40, 20, 50, 24)):
    return SimpleNamespace(text=text, box=box, score=.99)


class ShopPriceTests(unittest.TestCase):
    def test_located_number_may_have_outer_symbols_but_no_internal_repairs(self):
        for text, standard, expected in [('90"', 90, 90), ('400"', 400, 400),
                                         ('♡200', 200, 200), ('160', 200, 160), ('45', 90, 45)]:
            with self.subTest(text=text):
                self.assertEqual(payable_amount([row(text)], ROI, standard), (expected, [40, 20, 50, 24]))
        for text in ('m', 'I', '20O', '200100', '200I', '2 00', '140'):
            with self.subTest(text=text):
                self.assertEqual(payable_amount([row(text)], ROI, 200), (-1, None))

    def test_two_payable_values_do_not_guess_a_discount(self):
        for texts in (['200', '100'], ['100', '160']):
            self.assertEqual(payable_amount([row(text) for text in texts], ROI, 200), (-1, None))

    def test_whole_roi_fallback_and_unknown_item_type_stay_unknown(self):
        self.assertEqual(payable_amount([row('200', ROI)], ROI, 200), (-1, None))
        self.assertEqual(payable_amount([row('200')], ROI, None), (-1, None))
        self.assertIsNone(price_key('unknown_item', 5))
        self.assertEqual(price_key('potential_drink', 1), 'potential_drink')
        self.assertEqual(price_key('melody_of_pummel', 15), 'melody_15')

    def test_shelf_and_purchase_share_the_same_number_and_located_click_box(self):
        reader = ShopInteractor.__new__(ShopInteractor)
        reader.image = np.zeros((720, 1280, 3), np.uint8)
        reader.context = SimpleNamespace(tasker=SimpleNamespace(stopping=False),
            run_recognition=Mock(return_value=SimpleNamespace(all_results=[row('90"')],
                                                             filtered_results=[row('90"')])))
        handler = ShopHandler.__new__(ShopHandler)
        handler.interactor = reader
        handler.data = SimpleNamespace(params=SimpleNamespace(item_standard_prices={'melody_5': 90}))
        self.assertEqual(handler._get_item_price({'price_roi': ROI}, 'melody_of_pummel', 5), 90)
        argv = SimpleNamespace(image=reader.image, custom_recognition_param=json.dumps({
            'roi': ROI, 'price': 90, 'internal_name': 'melody_of_pummel', 'quantity': 5}))
        with patch('custom.climb_tower_neural.shop_ui.interactor.ShopInteractor', return_value=reader):
            result = ShopItemPrice().analyze(reader.context, argv)
        self.assertEqual(list(result.box), [40, 20, 50, 24])
        self.assertEqual(result.detail['price'], 90)
        self.assertEqual([call.args[0] for call in reader.context.run_recognition.call_args_list],
                         [PRICE_NODE, PRICE_NODE])

    @unittest.skipUnless(os.environ.get('STELLA_SHOP_OCR_REPLAY') == '1', 'Opt-in native price OCR replay')
    def test_native_blue_red_and_gray_price_ink(self):
        from PIL import Image
        from maa.controller import CustomController
        from maa.library import Library
        from maa.pipeline import JOCR
        from maa.resource import Resource
        from maa.tasker import Tasker
        import tempfile

        class Controller(CustomController):
            def connect(self): return True
            def request_uuid(self): return 'offline-shop-payable-ink'
            def screencap(self): return np.zeros((720, 1280, 3), np.uint8)
            def no_input(self, *args): return False
            start_app = stop_app = click = swipe = touch_down = touch_move = touch_up = no_input
            click_key = input_text = key_down = key_up = no_input

        Library.open(Library.framework_libpath.parent, agent_server=False)
        Tasker.set_log_dir(tempfile.mkdtemp(prefix='shop-price-replay-'))
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        resource = Resource()
        resource.use_cpu()
        self.assertTrue(resource.post_ocr_model(os.environ['STELLA_OCR_MODEL_DIR']).wait().status.succeeded)
        nodes = json.loads((ROOT / 'assets/resource/base/pipeline/climb_tower/climb_tower_neural_numbers.json').read_text(encoding='utf-8'))
        self.assertTrue(resource.override_pipeline(nodes))
        controller = Controller()
        self.assertTrue(controller.post_connection().wait().status.succeeded)
        tasker = Tasker()
        tasker.bind(resource, controller)
        atlas = np.array(Image.open(ROOT / 'tools/checks/fixtures/tower_replay/payable_prices_jp.png').convert('RGB'))[:, :, ::-1].copy()
        params = nodes[PRICE_NODE]['recognition']['param']
        expected = (160, 100, 200, 200, 72, 45, 90, 400)
        standards = (200, 200, 200, 200, 90, 90, 90, 400)
        for ink in ('original', 'red_recolor', 'no_payable_ink'):
            for index, (value, standard) in enumerate(zip(expected, standards)):
                with self.subTest(ink=ink, slot=index + 1):
                    image = atlas[index * 35:(index + 1) * 35].copy()
                    blue, green, red = image.astype(np.int16).transpose(2, 0, 1)
                    payable = ((blue >= 110) & (blue <= 170) & (green >= 50)
                               & (green <= 170) & (red >= 20) & (red <= 160))
                    # Controlled color variants retain the actual glyph shapes;
                    # the red variant is not a separate real-game screenshot.
                    if ink == 'red_recolor':
                        image[payable] = (75, 50, 210)
                    elif ink == 'no_payable_ink':
                        image[payable] = 255
                    roi = [0, 0, 110, 35]
                    job = tasker.post_recognition('OCR', JOCR(**dict(params, roi=roi)), image).wait()
                    detail = tasker.get_node_detail(job.get().node_id_list[0]).recognition
                    result = payable_amount(detail.filtered_results if detail else (), roi, standard)
                    self.assertEqual(result[0], -1 if ink == 'no_payable_ink' else value)


if __name__ == '__main__':
    unittest.main()
