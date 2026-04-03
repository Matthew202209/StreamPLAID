


class ModelsFactory:

    @staticmethod
    def build_model(config):
        model_name = config["model_config"]["model_name"]

        if model_name == "plaid_hdrc":
            from models.mrhp_1.main_model import PLAID_HDRC
            return PLAID_HDRC(config)

        if model_name == "plaid_hdrc_g":
            from models.mrhp_g.main_model import PLAID_HDRC
            return PLAID_HDRC(config)

        elif model_name == "mrhp_p_test":
            from models.mrhp_p_test.main_model import PLAID_P_TEST
            return PLAID_P_TEST(config)

        elif model_name == "ab_flat":
            from models.mrhp_ab_flat.main_model import AB_Flat_HDRC
            return AB_Flat_HDRC(config)

        elif model_name == "no_repair":
            from models.mrhp_ab_no_repair.main_model import No_Repair_HDRC
            return No_Repair_HDRC(config)

        else:
            raise ValueError(f"Model {model_name} is not supported.")


