import pytest
from .. import sjSDM_py as fa
import numpy as np
import torch


@pytest.fixture
def preserve_default_dtype():
    previous = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(previous)

@pytest.fixture
def data():
    def _get(a=5, b=5, c=100):
        return (np.random.randn(c,a), np.random.binomial(1,0.5,[c, b]))
    return _get

@pytest.fixture
def data_sp():
    def _get(a=5, b=5, c=100):
        return (np.random.randn(c,a), np.random.binomial(1,0.5,[c, b]), np.random.uniform(-1,1,[c,2]))
    return _get


@pytest.fixture
def model_base():
    def _get(inp=5,out=5,hidden=[], activation=['linear'],bias=[False], df=5,
             optimizer=fa.optimizer_adamax(1), l1_d = 0.0, l2_d = 0.0,
             l1_cov=0.0, l2_cov=0.0, link="logit",reg_on_Cov=True, reg_on_Diag=True,inverse=False, dropout=-99,
             device="cpu", dtype="float32"):
        model = fa.Model_sjSDM(device=device, dtype=dtype)
        model.add_env(input_shape=inp, output_shape=out, bias=bias,
                      hidden=hidden, activation=activation, l1=l1_d, l2=l2_d, dropout=dropout)
        model.build(df=df, l1=l1_cov,l2=l2_cov,optimizer=optimizer,link=link,reg_on_Cov=reg_on_Cov, reg_on_Diag=reg_on_Diag, inverse=inverse)
        return model
    return _get


def test_auto_float64_uses_torch(
    monkeypatch, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "auto")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    monkeypatch.setattr(
        mojo_bridge,
        "mojo_logit_loss",
        lambda *args: pytest.fail("float64 auto mode selected Mojo"),
    )
    model = model_base(inp=2, out=3, df=2, dtype="float64")
    mu = torch.zeros((4, 3), dtype=torch.float64)
    y = torch.zeros((4, 3), dtype=torch.float64)
    loss = model._loss_function(
        mu, y, model.sigma, 4, 5, 2, model.alpha, "cpu", model.dtype
    )
    assert loss.dtype == torch.float64


def test_auto_float32_uses_mojo(
    monkeypatch, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    calls = 0

    def fake_mojo(mu, y, sigma, sampling, alpha):
        nonlocal calls
        calls += 1
        return mu.sum(dim=1) * 0.0

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "auto")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    monkeypatch.setattr(mojo_bridge, "mojo_logit_loss", fake_mojo)
    model = model_base(inp=2, out=3, df=2, dtype="float32")
    model._loss_function(
        torch.zeros((4, 3)), torch.zeros((4, 3)), model.sigma,
        4, 5, 2, model.alpha, "cpu", model.dtype,
    )
    assert calls == 1


def test_forced_float64_reaches_strict_bridge_guard(
    monkeypatch, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "1")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    model = model_base(inp=2, out=3, df=2, dtype="float64")
    with pytest.raises(RuntimeError, match="CPU float32"):
        model._loss_function(
            torch.zeros((4, 3), dtype=torch.float64),
            torch.zeros((4, 3), dtype=torch.float64),
            model.sigma, 4, 5, 2, model.alpha, "cpu", model.dtype,
        )


def test_auto_missing_data_never_calls_mojo(
    monkeypatch, data, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "auto")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    monkeypatch.setattr(
        mojo_bridge,
        "mojo_logit_loss",
        lambda *args: pytest.fail("one batch selected Mojo in a missing-data fit"),
    )
    X, Y = data(a=2, b=3, c=20)
    Y = Y.astype(float)
    Y[0, 0] = np.nan
    model = model_base(inp=2, out=3, df=2)
    model.fit(X, Y, epochs=1, batch_size=10, verbose=False)
    model.logLik(X, Y, batch_size=10)


def test_forced_missing_data_fails_before_dataloader(
    monkeypatch, data, model_base, preserve_default_dtype
):
    from ..sjSDM_py import mojo_bridge

    monkeypatch.setenv("SJSDM_MOJO_BACKEND", "1")
    monkeypatch.setattr(mojo_bridge, "mojo_available", lambda: True)
    X, Y = data(a=2, b=3, c=20)
    Y = Y.astype(float)
    Y[0, 0] = np.nan
    model = model_base(inp=2, out=3, df=2)
    monkeypatch.setattr(
        model,
        "_get_DataLoader",
        lambda *args, **kwargs: pytest.fail("DataLoader constructed before guard"),
    )
    with pytest.raises(RuntimeError, match="missing responses"):
        model.fit(X, Y, epochs=1, batch_size=10, verbose=False)

@pytest.fixture
def model_base_sp():
    def _get(inp=5,out=5,hidden=[], activation=['linear'],bias=[False], df=5, 
            hidden_sp=[],activation_sp=['linear'],bias_sp=[False], sp_l1=0.0, sp_l2=0.0, 
             optimizer=fa.optimizer_adamax(1), l1_d = 0.0, l2_d = 0.0, 
             l1_cov=0.0, l2_cov=0.0, link="logit",reg_on_Cov=True, reg_on_Diag=True,inverse=False):
        model = fa.Model_sjSDM()
        model.add_env(input_shape=inp, output_shape=out, bias=bias,
                      hidden=hidden, activation=activation, l1=l1_d, l2=l2_d)
        model.add_spatial(input_shape=2, output_shape=out, hidden=hidden_sp,bias=bias_sp, activation=activation_sp,l1=sp_l1,l2=sp_l2)
        model.build(df=df, l1=l1_cov,l2=l2_cov,optimizer=optimizer,link=link,reg_on_Cov=reg_on_Cov, reg_on_Diag=reg_on_Diag, inverse=inverse)
        return model
    return _get


@pytest.mark.parametrize("inp,out,epochs,batch,early_stopping_training", 
                        [(5,5,1,10,-1), 
                         (3,10,2,2,-1),
                         (1,15,5,5,2),
                         (20,11,1, 100, 2),
                         pytest.param(5,5,-1,1,-1, marks=pytest.mark.xfail),
                         pytest.param(5,5,2,-1,-1, marks=pytest.mark.xfail),
                         pytest.param(0,5,2,1,-1, marks=pytest.mark.xfail),
                         pytest.param(5,0,2,1,-1, marks=pytest.mark.xfail)])
def test_base(data, model_base, inp, out, epochs, batch, early_stopping_training):
    X, Y = data(inp, out)
    model = model_base(inp, out)
    model.fit(X, Y, epochs=epochs, batch_size=batch, early_stopping_training=early_stopping_training)
    model.logLik(X, Y, batch_size=batch)
    pred = model.predict(X, batch_size=batch)
    model.se(X, Y, batch_size=batch)
    assert pred.shape[0] == X.shape[0]


@pytest.mark.parametrize("links", [("probit"), ("linear"), ("logit"), pytest.param("failed", marks=pytest.mark.xfail)])
def test_link(data, model_base, links):
    X, Y = data()
    model = model_base(link=links)
    model.fit(X, Y, epochs=1, batch_size=50)
    assert model.link == links
    model.logLik(X, Y)
    pred = model.predict(X)
    model.se(X, Y)
    assert pred.shape[0] == X.shape[0]


@pytest.mark.parametrize("l1_d,l2_d,l2_cov,l1_cov,reg_on_Cov,reg_on_Diag,inverse", 
                        [
                            (-1,0.1, -1, -1, True, True, False),
                            (0.1,-1, -1, -1, True, True, False),
                            (0.1,0.1, -1, -1, True, True, False),
                            (0.1,0.1, -1, 0.1, True, True, False),
                            (0.1,0.1, 0.1, -1, True, True, False),
                            (0.1,0.1, 0.1, 0.1, True, True, False),
                            (0.1,0.1, 0.1, 0.1, False, True, False),
                            (0.1,0.1, 0.1, 0.1, True, False, False),
                            (0.1,0.1, 0.1, 0.1, True, False, True)
                        ])
def test_regularization(data, model_base, l1_d, l2_d, l2_cov, l1_cov, reg_on_Cov, reg_on_Diag,inverse):
    X, Y = data()
    model = model_base(5, 5, l1_d=l1_d, l2_d=l2_d, l2_cov=l2_cov, l1_cov=l1_cov, reg_on_Cov=reg_on_Cov, reg_on_Diag=reg_on_Diag, inverse=inverse)
    model.fit(X, Y, epochs=1, batch_size=10)
    model.logLik(X, Y, batch_size=10)
    pred = model.predict(X, batch_size=10)
    model.se(X, Y)
    assert pred.shape[0] == X.shape[0]

@pytest.mark.parametrize("inp,out,hidden,activation,bias,dropout", 
                        [
                            (5,5,[], ['linear'],[False], 0.1),
                            (3,3,[5,5,5], ['linear'],[True], 0.1),
                            (2,10,[3,3,3], ['relu'], [True,False,True,True], -1),
                            (2,10,[3,3,3], ['sigmoid'],[True], -1),
                            (2,10,[3,3,3], ['tanh'], [True], -1),
                            (2,10,[3,3,3], ['selu'], [True], -1),
                            (2,10,[3,3,3], ['leakyrelu'], [True], -1),
                            (2,10,[3,3,3], ['relu', 'tanh', 'sigmoid'],[True, False, True, True], 0.3)
                        ])
def test_dnn(data, model_base, inp, out, hidden, activation,bias, dropout):
    X, Y = data(inp, out)
    model = model_base(inp, out, hidden=hidden, activation=activation, dropout=dropout)
    model.fit(X, Y, epochs=1, batch_size=50)
    model.logLik(X, Y)
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]



@pytest.mark.parametrize("inp,out,epochs,batch", 
                        [(5,5,1,10), 
                         (3,10,2,2),
                         (1,15,5,5),
                         (20,11,1, 100),
                         pytest.param(5,5,-1,1, marks=pytest.mark.xfail),
                         pytest.param(5,5,2,-1, marks=pytest.mark.xfail),
                         pytest.param(0,5,2,1, marks=pytest.mark.xfail),
                         pytest.param(5,0,2,1, marks=pytest.mark.xfail)])
def test_base_sp(data_sp, model_base_sp, inp, out, epochs, batch):
    X, Y, SP = data_sp(inp, out)
    model = model_base_sp(inp, out)
    model.fit(X, Y, epochs=epochs, batch_size=batch)
    model.logLik(X, Y,SP=SP, batch_size=batch)
    pred = model.predict(X, SP=SP, batch_size=batch)
    model.se(X, Y, SP=SP, batch_size=batch)
    assert pred.shape[0] == X.shape[0]


@pytest.mark.parametrize("inp,out,hidden,activation,l1,l2", 
                        [
                            (5,5,[], ['linear'], -1,-1),
                            (3,3,[5,5,5], ['linear'],-1,-1),
                            (2,10,[3,3,3], ['relu'],0.1,-1),
                            (2,10,[3,3,3], ['sigmoid'], -1, 0.1),
                            (2,10,[3,3,3], ['tanh'], 0.1, 0.1),
                            (2,10,[3,3,3], ['relu', 'tanh', 'sigmoid'], 0.5, 0.5)
                        ])
def test_dnn_sp(data_sp, model_base_sp, inp, out, hidden, activation,l1,l2):
    X, Y, SP = data_sp(inp, out)
    model = model_base_sp(inp, out, hidden_sp=hidden, activation_sp=activation,sp_l1=l1,sp_l2=l2)
    model.fit(X, Y, SP=SP, epochs=1, batch_size=50)
    model.logLik(X, Y, SP=SP)
    pred = model.predict(X, SP=SP)
    assert pred.shape[0] == X.shape[0]


def test_base_weights(data_sp, model_base_sp):
    epochs = 1
    batch = 5
    X, Y, SP = data_sp()
    model = model_base_sp(5, 5, hidden = [5,5,5], hidden_sp=[3,3,3])
    model.fit(X, Y, epochs=epochs, batch_size=batch)

    model.set_env_weights(model.env_weights)
    model.set_spatial_weights(model.spatial_weights)

    ## still buggy
    #model.logLik(X, Y,SP=SP, batch_size=batch)
    #pred = model.predict(X, SP=SP, batch_size=batch)
    #model.se(X, Y, SP=SP, batch_size=batch)
    #assert pred.shape[0] == X.shape[0]